"""Stage 2 -- fetch: inputs/manifest.csv -> temp/filings/.

    python -m tools.fetch
    python -m tools.fetch --dry-run
    python -m tools.fetch --limit 3
    python -m tools.fetch --set fetch.concurrency=8 --set fetch.rate_limit_per_sec=6
    python -m tools.fetch --force

Downloads each filing's primary document as HTML or text, named by accession.

RE-RUNNABILITY IS THE POINT
===========================

This stage reads the manifest stage 1 left on disk and never re-queries the
submissions API (Rule 1). Within itself it is idempotent: every accession already
recorded in the ledger as ``ok``, with its file still present at the recorded
size, is skipped BEFORE the request is constructed. A re-run over an unchanged
manifest therefore issues zero HTTP requests, and the run log's ``requests: 0``
is the proof that the skip check really precedes the request rather than
discarding a response after paying for it.

The ledger is a separate file from the documents because a directory listing
cannot distinguish "fetched successfully" from "half-written when the process
was killed". Documents are written atomically for the same reason: a truncated
file that happened to match an expected size would otherwise be skipped forever.

NO PDF PATH
===========

EDGAR serves primary documents as HTML or text. Anything else is logged as an
error and skipped rather than parsed -- adding a PDF path would be pure overhead
for a corpus that never contains one.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tools.config import Config, ConfigError, add_config_args, load_config
from tools.edgar import (
    DOC_TYPE_EXHIBIT,
    DOC_TYPE_PRIMARY,
    LEDGER_FIELDS,
    LEGACY_LEDGER_FIELDS,
    EdgarClient,
    EdgarError,
    as_stored_path,
    from_stored_path,
    manifest_key,
    read_manifest,
    stored_filename,
)
from tools.runlog import RunLog, StageRecorder

__all__ = ["fetch"]


def _load_ledger(
    path: Path, manifest: list[dict[str, str]]
) -> tuple[dict[tuple[str, str], dict[str, str]], int]:
    """Read the fetch ledger keyed by ``(accession, document)``.

    Returns ``(ledger, migrated_row_count)``. A missing file is an empty ledger.

    LEGACY MIGRATION
    ----------------
    Ledgers written before exhibits existed have a 7-column header and no
    ``document``. Every row in such a file is a primary document -- not by
    assumption, but because fetching ``primary_doc_url`` was the only thing
    stage 2 could do. Each legacy row is therefore matched to the manifest row
    for its accession with ``doc_type == primary``, and takes that row's
    ``document`` value.

    The filename is taken from the manifest rather than derived from our own
    on-disk name, so the value recorded is the one EDGAR published. A legacy row
    whose accession has no primary manifest row keeps ``document`` empty and is
    returned unmatched for the caller to log -- it is never guessed.
    """
    if not path.is_file():
        return {}, 0

    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        header = tuple(reader.fieldnames or ())
        rows = [dict(row) for row in reader]

    if header == LEDGER_FIELDS:
        # Later rows win: the ledger is append-only, so the last entry for a
        # document is its current state.
        return {(r["accession"], r.get("document", "")): r for r in rows}, 0

    if header != LEGACY_LEDGER_FIELDS:
        raise EdgarError(
            f"{path} has header {header!r}, expected {LEDGER_FIELDS!r} "
            f"(or the legacy {LEGACY_LEDGER_FIELDS!r}). "
            f"Delete it to re-fetch, or fix it by hand."
        )

    primary_document = {
        r["accession"]: r.get("document", "")
        for r in manifest
        if r.get("doc_type") == DOC_TYPE_PRIMARY
    }
    ledger: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        document = primary_document.get(row["accession"], "")
        upgraded = dict(row)
        upgraded["document"] = document
        upgraded["doc_type"] = DOC_TYPE_PRIMARY
        ledger[(row["accession"], document)] = upgraded
    return ledger, len(rows)


def _rewrite_ledger(path: Path, ledger: dict[tuple[str, str], dict[str, str]]) -> None:
    """Rewrite the whole ledger in the current format, atomically.

    Used once, after a legacy ledger is migrated, so later runs read the new
    header directly instead of re-migrating every time.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(LEDGER_FIELDS), lineterminator="\n")
        writer.writeheader()
        for row in ledger.values():
            writer.writerow({k: row.get(k, "") for k in LEDGER_FIELDS})
    tmp.replace(path)


def _append_ledger(path: Path, rows: list[dict[str, Any]]) -> None:
    """Append rows to the ledger, writing the header if the file is new."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.is_file()
    with path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(LEDGER_FIELDS), lineterminator="\n")
        if is_new:
            writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in LEDGER_FIELDS})


def _already_fetched(
    key: tuple[str, str], ledger: dict[tuple[str, str], dict[str, str]]
) -> Path | None:
    """Return the existing path if this document is complete on disk, else None.

    Keyed by ``(accession, document)``: a filing now contributes several
    documents, so the accession alone no longer identifies what was fetched.

    Checks the file as well as the ledger. A ledger entry whose document was
    deleted, or whose size no longer matches what was recorded, is not a
    successful fetch and must not cause a skip.
    """
    entry = ledger.get(key)
    if not entry or entry.get("status") != "ok":
        return None
    path = from_stored_path(entry.get("path", ""))
    if not path.is_file():
        return None
    try:
        if int(entry.get("bytes") or -1) != path.stat().st_size:
            return None
    except ValueError:
        return None
    return path


def _extension(content_type: str) -> str:
    return ".txt" if content_type.split(";")[0].strip() == "text/plain" else ".html"


def _write_atomic(path: Path, payload: bytes) -> None:
    """Write bytes via a temp file and rename.

    A process killed mid-write must not leave a truncated document behind: the
    next run's skip check would see a plausible file and never repair it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_bytes(payload)
    tmp.replace(path)


def _fetch_one(
    row: dict[str, str],
    client: EdgarClient,
    config: Config,
    stage: StageRecorder,
    out_dir: Path,
    claimed: dict[str, tuple[str, str]],
    claimed_lock: threading.Lock,
) -> dict[str, Any]:
    """Download one document. Returns a ledger row; never raises."""
    accession = row["accession"]
    document = (row.get("document") or "").strip()
    doc_type = row.get("doc_type") or DOC_TYPE_PRIMARY
    url = (row.get("doc_url") or "").strip()
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    def ledger_row(status: str, **extra: Any) -> dict[str, Any]:
        return {
            "accession": accession,
            "document": document,
            "doc_type": doc_type,
            "path": "",
            "status": status,
            "bytes": 0,
            "content_type": "",
            "sha256": "",
            "fetched_at": now,
            **extra,
        }

    if not url:
        # Stage 1 already logged why this is null. Re-logging keeps the failure
        # visible in this stage's own error count rather than making a reader
        # correlate two run logs to find out why a document is absent.
        stage.error(
            "manifest row has no doc_url; nothing to fetch",
            context={
                "accession": accession,
                "document": document,
                "ticker": row.get("ticker"),
            },
        )
        return ledger_row("no_url")

    accepted = [str(t).lower() for t in config.get("fetch.accept_content_types")]

    try:
        response = client.get(url)
        content_type = response.headers.get("content-type", "")
        base_type = content_type.split(";")[0].strip().lower()
        if base_type and base_type not in accepted:
            raise EdgarError(
                f"unexpected content-type {base_type!r} for {accession}/{document} "
                f"(accepted: {accepted}). Not written."
            )
        payload = response.content
        name = stored_filename(accession, doc_type, document, _extension(base_type))

        # Two EDGAR documents in one filing can sanitize to the same on-disk
        # name. Writing both would silently leave one document holding the
        # other's bytes, which no later stage could detect.
        with claimed_lock:
            owner = claimed.get(name)
            if owner is not None and owner != (accession, document):
                raise EdgarError(
                    f"on-disk name {name!r} is already claimed by "
                    f"{owner[0]}/{owner[1]}; refusing to overwrite it with "
                    f"{accession}/{document}"
                )
            claimed[name] = (accession, document)

        path = out_dir / name
        _write_atomic(path, payload)
        stage.count()
        stage.bytes_out(len(payload))
        return ledger_row(
            "ok",
            path=as_stored_path(path),
            bytes=len(payload),
            content_type=base_type,
            # Cheap here, and it lets a stage-3 re-parse be verified against the
            # exact bytes that were fetched rather than whatever is on disk now.
            sha256=hashlib.sha256(payload).hexdigest(),
        )
    except (EdgarError, OSError) as exc:
        stage.error(
            exc,
            context={"accession": accession, "document": document, "url": url},
        )
        return ledger_row("error")


def fetch(
    config_path: str,
    overrides: list[str],
    *,
    dry_run: bool = False,
    limit: int | None = None,
    force: bool = False,
) -> int:
    config = load_config(config_path, overrides=overrides)

    manifest_path = Path(config.get("discover.out_path"))
    out_dir = Path(config.get("fetch.out_dir"))
    ledger_path = Path(config.get("fetch.ledger_path"))

    manifest = read_manifest(manifest_path)
    if limit is not None:
        manifest = manifest[:limit]

    # The full manifest, not the --limit slice, is what the legacy ledger is
    # matched against: a limited run must not migrate rows it cannot see.
    full_manifest = read_manifest(manifest_path) if limit is not None else manifest
    if force:
        ledger, migrated = {}, 0
    else:
        ledger, migrated = _load_ledger(ledger_path, full_manifest)
    unmatched_legacy = [k for k in ledger if not k[1]] if migrated else []

    pending: list[dict[str, str]] = []
    skipped = 0
    for row in manifest:
        # The skip decision happens here, before any request object exists.
        if _already_fetched(manifest_key(row), ledger) is not None:
            skipped += 1
            continue
        pending.append(row)

    by_type = Counter(r.get("doc_type") or DOC_TYPE_PRIMARY for r in pending)

    if dry_run:
        print("DRY RUN -- no request made, no file written.")
        print(f"  manifest    {manifest_path} ({len(manifest)} document rows)")
        print(f"  ledger      {ledger_path} ({len(ledger)} entries)"
              + (f", {migrated} legacy row(s) would migrate" if migrated else ""))
        print(f"  would skip  {skipped} already fetched")
        print(f"  would GET   {len(pending)} document(s) "
              f"({by_type.get(DOC_TYPE_PRIMARY, 0)} primary, "
              f"{by_type.get(DOC_TYPE_EXHIBIT, 0)} exhibit)")
        print(f"  concurrency {config.get('fetch.concurrency')} "
              f"@ {config.get('fetch.rate_limit_per_sec')} req/sec")
        print(f"  would write {out_dir}/<accession>[-<document>].html|.txt")
        for row in pending[:10]:
            print(f"    GET {row['doc_url'] or '(no url)'}")
        if len(pending) > 10:
            print(f"    ... and {len(pending) - 10} more")
        return 0

    results: list[dict[str, Any]] = []

    with RunLog(config) as log:
        with log.stage("fetch") as stage:
            stage.note(
                manifest_path=str(manifest_path),
                manifest_rows=len(manifest),
                concurrency=config.get("fetch.concurrency"),
                rate_limit_per_sec=config.get("fetch.rate_limit_per_sec"),
                forced=force,
            )
            out_dir.mkdir(parents=True, exist_ok=True)

            if migrated:
                # Recorded rather than done silently: it changes the ledger's
                # schema on disk, and the count is what shows the migration
                # preserved every previously fetched document.
                stage.note(ledger_migrated_rows=migrated)
                if unmatched_legacy:
                    for accession, _ in unmatched_legacy:
                        stage.error(
                            "legacy ledger row has no matching primary manifest "
                            "row; its document name is unknown and is left blank",
                            context={"accession": accession},
                        )
                    stage.note(ledger_migrated_unmatched=len(unmatched_legacy))
                _rewrite_ledger(ledger_path, ledger)

            # Names claimed this run, plus those already on disk, so a collision
            # with an existing file is caught as well as one within this run.
            claimed: dict[str, tuple[str, str]] = {
                from_stored_path(r["path"]).name: (r["accession"], r.get("document", ""))
                for r in ledger.values()
                if r.get("status") == "ok" and r.get("path")
            }
            claimed_lock = threading.Lock()

            if pending:
                with EdgarClient(config, stage) as client:
                    workers = max(1, int(config.get("fetch.concurrency", 4)))
                    with ThreadPoolExecutor(max_workers=workers) as pool:
                        results = list(
                            pool.map(
                                lambda row: _fetch_one(
                                    row, client, config, stage, out_dir,
                                    claimed, claimed_lock,
                                ),
                                pending,
                            )
                        )
                throttle_wait = client.throttle_wait_sec
            else:
                throttle_wait = 0.0

            _append_ledger(ledger_path, results)

            ok = sum(1 for r in results if r["status"] == "ok")
            failed = sum(1 for r in results if r["status"] == "error")
            no_url = sum(1 for r in results if r["status"] == "no_url")
            ok_by_type = Counter(
                r["doc_type"] for r in results if r["status"] == "ok"
            )
            stage.note(
                fetched=ok,
                fetched_primary=ok_by_type.get(DOC_TYPE_PRIMARY, 0),
                fetched_exhibit=ok_by_type.get(DOC_TYPE_EXHIBIT, 0),
                skipped=skipped,
                failed=failed,
                no_url=no_url,
                ledger_path=str(ledger_path),
                out_dir=str(out_dir),
                throttle_wait_sec=throttle_wait,
            )

    print(f"Fetched {ok} document(s) into {out_dir}")
    if ok:
        print(f"  {ok_by_type.get(DOC_TYPE_PRIMARY, 0)} primary, "
              f"{ok_by_type.get(DOC_TYPE_EXHIBIT, 0)} exhibit")
    print(f"  skipped {skipped} already present, {failed} failed, {no_url} with no URL")
    if migrated:
        print(f"  migrated {migrated} legacy ledger row(s) to the per-document schema")
    print(f"  ledger: {ledger_path}")
    if failed:
        print(f"ERROR: {failed} document(s) failed after retries.", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tools.fetch",
        description="Stage 2: download each manifest filing's primary document.",
    )
    add_config_args(parser)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print what would be fetched and skipped; make no request",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="only process the first N manifest rows",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="ignore the ledger and re-fetch every document",
    )
    args = parser.parse_args(argv)

    try:
        return fetch(
            args.config,
            args.overrides,
            dry_run=args.dry_run,
            limit=args.limit,
            force=args.force,
        )
    except (ConfigError, EdgarError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
