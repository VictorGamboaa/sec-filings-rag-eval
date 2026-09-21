"""Stage 1 -- discover: universe.csv -> inputs/manifest.csv.

    python -m tools.discover
    python -m tools.discover --dry-run
    python -m tools.discover --set discover.date_from=2025-01-01
    python -m tools.discover --set discover.forms='["10-Q"]'

Queries data.sec.gov's submissions API once per CIK and writes one manifest row
per filing whose FILING DATE falls inside the configured window.

SCOPE IS A FILING-DATE WINDOW, NEVER A FISCAL LABEL
===================================================

The five names in this universe do not share a fiscal calendar -- Albertsons
runs an offset fiscal year -- so selecting by a label like "Q2 FY25" would pull
different real periods for different companies and make any cross-name retrieval
comparison meaningless. Selection is therefore purely on ``filing_date`` against
``discover.date_from`` / ``discover.date_to``. ``period_of_report`` is carried
through to the manifest as METADATA ONLY and is never used to include or exclude
a filing.

Narrowing the window is also how the scaling curve gets produced, which is the
second reason it has to be config rather than anything baked in.

RE-RUNNABILITY
==============

Output is written deterministically (sorted, atomic), so re-running over an
unchanged window produces a byte-identical manifest and "did the corpus change?"
is answerable by diff. Stage 2 reads the file this stage leaves on disk and never
re-queries submissions.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from tools.config import Config, ConfigError, add_config_args, load_config
from tools.edgar import (
    DOC_TYPE_EXHIBIT,
    DOC_TYPE_PRIMARY,
    EdgarClient,
    EdgarError,
    accession_dashed,
    filing_index_url,
    iter_recent_filings,
    iter_shard_filings,
    load_universe,
    manifest_key,
    matches_exhibit_type,
    parse_index_headers,
    primary_doc_url,
    write_manifest,
)
from tools.runlog import RunLog, StageRecorder

__all__ = ["discover"]


def _parse_date(value: str, label: str) -> date:
    try:
        return date.fromisoformat(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ConfigError(
            f"{label} must be an ISO date (YYYY-MM-DD), got {value!r}"
        ) from exc


def _resolve_window(config: Config) -> tuple[date, date]:
    """Resolve the filing-date window, with ``date_to: null`` meaning today.

    The resolved upper bound is returned (and noted in the run log) rather than
    left implicit, so a run log read months later still says exactly which
    window produced its numbers.
    """
    date_from = _parse_date(config.get("discover.date_from"), "discover.date_from")
    raw_to = config.get("discover.date_to", None)
    date_to = (
        datetime.now(timezone.utc).date()
        if raw_to is None
        else _parse_date(raw_to, "discover.date_to")
    )
    if date_to < date_from:
        raise ConfigError(
            f"empty window: discover.date_to ({date_to}) precedes "
            f"discover.date_from ({date_from})"
        )
    return date_from, date_to


def _row_from_filing(
    filing: dict[str, Any],
    entry: dict[str, str],
    archives_url: str,
    stage: StageRecorder,
) -> dict[str, Any]:
    """Build one manifest row from one submissions record."""
    accession = accession_dashed(filing["accessionNumber"])
    document = str(filing.get("primaryDocument") or "").strip()

    if document:
        url = primary_doc_url(archives_url, entry["cik"], accession, document)
    else:
        # Rule 4: a missing field is null and logged. The filename cannot be
        # reconstructed without guessing, and a guessed URL would 404 at fetch
        # time while looking like real data in the manifest.
        url = None
        stage.error(
            "submissions record has no primaryDocument; primary_doc_url is null",
            context={
                "cik": entry["cik"],
                "ticker": entry["ticker"],
                "accession": accession,
                "form": filing.get("form"),
            },
        )

    period = str(filing.get("reportDate") or "").strip() or None
    if period is None:
        stage.error(
            "submissions record has no reportDate; period_of_report is null",
            context={"accession": accession, "form": filing.get("form")},
        )

    return {
        "cik": entry["cik"],
        "ticker": entry["ticker"],
        "accession": accession,
        "form": str(filing.get("form") or "").strip(),
        "filing_date": str(filing.get("filingDate") or "").strip() or None,
        "period_of_report": period,
        "doc_type": DOC_TYPE_PRIMARY,
        "exhibit_label": None,
        "document": document or None,
        "doc_url": url,
    }


def _exhibit_rows(
    primary_row: dict[str, Any],
    client: EdgarClient,
    config: Config,
    stage: StageRecorder,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Fetch a filing's index-headers and build a row per qualifying exhibit.

    Returns ``(rows, types_seen)``. Raises EdgarError if the index could not be
    fetched or parsed -- the caller counts that separately from "this filing has
    no qualifying exhibit", because conflating the two would let a network
    failure masquerade as a structural fact about the corpus.
    """
    archives_url = config.get("fetch.archives_url")
    prefixes = list(config.get("discover.exhibit_types", []))

    url = filing_index_url(archives_url, primary_row["cik"], primary_row["accession"])
    documents = parse_index_headers(client.get(url).text)
    if not documents:
        raise EdgarError(f"no <DOCUMENT> blocks parsed from {url}")

    types_seen = [str(d.get("type") or "") for d in documents]
    rows: list[dict[str, Any]] = []

    for doc in documents:
        if not matches_exhibit_type(doc.get("type"), prefixes):
            continue
        filename = doc.get("filename")
        if not filename:
            # Rule 4: never reconstruct a filename. A guessed one would 404 at
            # fetch time while looking like real data in the manifest.
            stage.error(
                "index-headers DOCUMENT block has no FILENAME; exhibit skipped",
                context={
                    "accession": primary_row["accession"],
                    "ticker": primary_row["ticker"],
                    "type": doc.get("type"),
                    "sequence": doc.get("sequence"),
                },
            )
            continue
        rows.append(
            {
                **{
                    k: primary_row[k]
                    for k in (
                        "cik",
                        "ticker",
                        "accession",
                        "form",
                        "filing_date",
                        "period_of_report",
                    )
                },
                "doc_type": DOC_TYPE_EXHIBIT,
                "exhibit_label": str(doc["type"]).strip(),
                "document": filename,
                "doc_url": primary_doc_url(
                    archives_url, primary_row["cik"], primary_row["accession"], filename
                ),
            }
        )
    return rows, types_seen


def _discover_one(
    entry: dict[str, str],
    client: EdgarClient,
    config: Config,
    stage: StageRecorder,
    window: tuple[date, date],
) -> dict[str, Any]:
    """Fetch and filter one CIK's submissions. Returns rows plus shard stats."""
    date_from, date_to = window
    submissions_url = str(config.get("discover.submissions_url")).rstrip("/")
    archives_url = config.get("fetch.archives_url")
    forms = set(config.get("discover.forms"))
    follow_shards = bool(config.get("discover.follow_older_shards", True))

    url = f"{submissions_url}/CIK{entry['cik']}.json"
    submissions = client.get_json(url)

    rows: list[dict[str, Any]] = []
    seen_dates: list[str] = []
    considered = 0

    def take(filing: dict[str, Any]) -> None:
        nonlocal considered
        considered += 1
        filing_date = str(filing.get("filingDate") or "").strip()
        if filing_date:
            seen_dates.append(filing_date)
        # Exact form match: "10-Q" must not sweep in "10-Q/A" unless the
        # configured list names it. An amendment is a different document with
        # different content, and silently mixing the two would confound any
        # retrieval measurement taken over the corpus.
        if str(filing.get("form") or "").strip() not in forms:
            return
        if not filing_date:
            stage.error(
                "submissions record has no filingDate; cannot place it in the "
                "window, so it is excluded",
                context={"cik": entry["cik"], "accession": filing.get("accessionNumber")},
            )
            return
        if not (date_from <= _parse_date(filing_date, "filingDate") <= date_to):
            return
        rows.append(_row_from_filing(filing, entry, archives_url, stage))

    for filing in iter_recent_filings(submissions):
        take(filing)

    # filings.recent covers roughly the last year or 1000 filings. If the window
    # starts before the oldest record in it, the window is NOT fully covered and
    # a manifest built from recent alone would be quietly short -- which would
    # look like "this filer files less" rather than "we did not ask for it all".
    oldest_recent = min(seen_dates) if seen_dates else None
    needs_shards = oldest_recent is None or date_from < _parse_date(
        oldest_recent, "filingDate"
    )
    shards_fetched = 0
    truncated = False

    if needs_shards:
        shard_files = submissions.get("filings", {}).get("files", []) or []
        if not follow_shards:
            if shard_files:
                truncated = True
                stage.error(
                    "window predates filings.recent and "
                    "discover.follow_older_shards is false: manifest is truncated",
                    context={
                        "cik": entry["cik"],
                        "ticker": entry["ticker"],
                        "oldest_in_recent": oldest_recent,
                        "date_from": str(date_from),
                        "shards_available": len(shard_files),
                    },
                )
        else:
            for shard in shard_files:
                name = str(shard.get("name") or "").strip()
                if not name:
                    continue
                if not _shard_overlaps(shard, date_from, date_to):
                    continue
                try:
                    payload = client.get_json(f"{submissions_url}/{name}")
                except EdgarError as exc:
                    truncated = True
                    stage.error(
                        exc,
                        context={
                            "cik": entry["cik"],
                            "shard": name,
                            "note": "older filings shard unavailable; manifest "
                            "may be truncated for this CIK",
                        },
                    )
                    continue
                shards_fetched += 1
                for filing in iter_shard_filings(payload):
                    take(filing)

    # De-duplicate primaries before indexing: a filing present in both recent and
    # an abutting shard must not cost two index requests.
    unique: dict[str, dict[str, Any]] = {}
    for row in rows:
        unique.setdefault(row["accession"], row)
    rows = list(unique.values())

    # --- exhibits --------------------------------------------------------
    # One index-headers request per filing whose form is configured for exhibit
    # expansion. Every such filing is indexed, because whether it HAS a
    # qualifying exhibit is precisely what the index answers -- and stage 1 must
    # not depend on stage 2's fetched documents to find out.
    exhibit_forms = set(config.get("discover.exhibit_forms", []))
    exhibit_rows: list[dict[str, Any]] = []
    types_seen: Counter[str] = Counter()
    indexed = 0
    with_exhibit = 0
    without_exhibit = 0
    index_failed = 0

    for row in rows:
        if row["form"] not in exhibit_forms:
            continue
        try:
            found, seen = _exhibit_rows(row, client, config, stage)
        except EdgarError as exc:
            # Counted separately from "no qualifying exhibit" on purpose: a
            # failed request is not evidence that a filing has no exhibit, and
            # conflating them would corrupt the tripwire that checks the
            # with/without split against an independent count.
            index_failed += 1
            stage.error(
                exc,
                context={
                    "cik": row["cik"],
                    "ticker": row["ticker"],
                    "accession": row["accession"],
                    "note": "filing index unavailable; exhibits not enumerated "
                    "for this filing",
                },
            )
            continue
        indexed += 1
        types_seen.update(t for t in seen if t)
        if found:
            with_exhibit += 1
            exhibit_rows.extend(found)
        else:
            without_exhibit += 1

    rows.extend(exhibit_rows)

    return {
        "rows": rows,
        "considered": considered,
        "shards_fetched": shards_fetched,
        "truncated": truncated,
        "oldest_in_recent": oldest_recent,
        "indexed": indexed,
        "with_exhibit": with_exhibit,
        "without_exhibit": without_exhibit,
        "index_failed": index_failed,
        "exhibit_rows": len(exhibit_rows),
        "types_seen": types_seen,
    }


def _shard_overlaps(shard: dict[str, Any], date_from: date, date_to: date) -> bool:
    """Does an older-filings shard's advertised range overlap the window?

    A shard whose range cannot be read is treated as overlapping and fetched.
    Skipping it would silently truncate the manifest, and an extra request costs
    far less than a corpus that is quietly missing filings.
    """
    raw_from = str(shard.get("filingFrom") or "").strip()
    raw_to = str(shard.get("filingTo") or "").strip()
    if not raw_from or not raw_to:
        return True
    try:
        shard_from = date.fromisoformat(raw_from)
        shard_to = date.fromisoformat(raw_to)
    except ValueError:
        return True
    return shard_from <= date_to and shard_to >= date_from


def discover(
    config_path: str,
    overrides: list[str],
    *,
    dry_run: bool = False,
    limit: int | None = None,
) -> int:
    config = load_config(config_path, overrides=overrides)

    date_from, date_to = _resolve_window(config)
    universe_path = Path(config.get("universe.out_path"))
    out_path = Path(config.get("discover.out_path"))
    submissions_url = str(config.get("discover.submissions_url")).rstrip("/")
    forms = list(config.get("discover.forms"))

    entries = load_universe(universe_path)
    if limit is not None:
        entries = entries[:limit]

    if dry_run:
        print("DRY RUN -- no request made, no file written.")
        print(f"  window      {date_from} .. {date_to}  (filing date, inclusive)")
        print(f"  forms       {forms}  (matched exactly)")
        print(f"  concurrency {config.get('discover.concurrency')} "
              f"@ {config.get('fetch.rate_limit_per_sec')} req/sec")
        print(f"  would write {out_path}")
        print(f"  {len(entries)} planned submissions request(s):")
        for entry in entries:
            print(f"    GET {submissions_url}/CIK{entry['cik']}.json   # {entry['ticker']}")
        print("  plus any older filings shards needed to cover the window,")
        print(
            f"  plus one filing-index request per {config.get('discover.exhibit_forms')} "
            f"filing found, to enumerate {config.get('discover.exhibit_types')} exhibits."
        )
        return 0

    failed: list[str] = []
    rows: list[dict[str, Any]] = []

    with RunLog(config) as log:
        with log.stage("discover") as stage:
            stage.note(
                date_from=str(date_from),
                date_to_effective=str(date_to),
                date_to_configured=config.get("discover.date_to", None),
                forms=forms,
                ciks=len(entries),
                concurrency=config.get("discover.concurrency"),
                rate_limit_per_sec=config.get("fetch.rate_limit_per_sec"),
            )

            with EdgarClient(config, stage) as client:

                def work(entry: dict[str, str]) -> tuple[dict[str, str], Any]:
                    try:
                        return entry, _discover_one(
                            entry, client, config, stage, (date_from, date_to)
                        )
                    except (EdgarError, ConfigError, KeyError, TypeError) as exc:
                        # One CIK failing must not cost the other four their
                        # manifest rows; the failure is recorded and the run
                        # exits non-zero so a partial manifest is never mistaken
                        # for a complete one.
                        stage.error(
                            exc,
                            context={"cik": entry["cik"], "ticker": entry["ticker"]},
                        )
                        return entry, None

                workers = max(1, int(config.get("discover.concurrency", 2)))
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    results = list(pool.map(work, entries))

            shards_fetched = 0
            truncated_ciks: list[str] = []
            per_ticker: Counter[str] = Counter()
            indexed = with_exhibit = without_exhibit = index_failed = 0
            types_seen: Counter[str] = Counter()

            for entry, result in results:
                if result is None:
                    failed.append(entry["ticker"])
                    continue
                rows.extend(result["rows"])
                shards_fetched += result["shards_fetched"]
                indexed += result["indexed"]
                with_exhibit += result["with_exhibit"]
                without_exhibit += result["without_exhibit"]
                index_failed += result["index_failed"]
                types_seen.update(result["types_seen"])
                if result["truncated"]:
                    truncated_ciks.append(entry["ticker"])
                per_ticker[entry["ticker"]] = len(result["rows"])
                if not result["rows"]:
                    stage.error(
                        "no filings matched the window for this CIK",
                        context={
                            "cik": entry["cik"],
                            "ticker": entry["ticker"],
                            "window": f"{date_from}..{date_to}",
                            "forms": forms,
                        },
                    )

            # De-duplicate on (accession, document). A filing can appear in both
            # recent and an older shard where their ranges abut, and a filing now
            # contributes several rows, so accession alone is no longer identity.
            deduped: dict[tuple[str, str], dict[str, Any]] = {}
            duplicates = 0
            for row in rows:
                key = manifest_key(row)
                if key in deduped:
                    duplicates += 1
                    continue
                deduped[key] = row
            rows = list(deduped.values())

            primaries = [r for r in rows if r["doc_type"] == DOC_TYPE_PRIMARY]
            exhibits = [r for r in rows if r["doc_type"] == DOC_TYPE_EXHIBIT]

            size = write_manifest(out_path, rows)
            stage.count(len(rows))
            stage.bytes_out(size)
            stage.note(
                manifest_rows=len(rows),
                out_path=str(out_path),
                filings=len(primaries),
                filings_by_form=dict(Counter(r["form"] for r in primaries)),
                filings_by_ticker=dict(per_ticker),
                duplicates_dropped=duplicates,
                shards_fetched=shards_fetched,
                window_truncated=bool(truncated_ciks),
                truncated_ciks=truncated_ciks,
                failed_ciks=failed,
                missing_primary_doc=sum(1 for r in primaries if not r["doc_url"]),
                # --- exhibit expansion, and the tripwire -------------------
                exhibit_forms=list(config.get("discover.exhibit_forms", [])),
                exhibit_types=list(config.get("discover.exhibit_types", [])),
                exhibit_index_fetched=indexed,
                filings_with_qualifying_exhibit=with_exhibit,
                # Expect roughly 70 against the current corpus. Much higher means
                # exhibit_types is failing to match, not that the filings are
                # exhibit-free -- read exhibit_types_seen to find out which.
                filings_without_qualifying_exhibit=without_exhibit,
                exhibit_index_failed=index_failed,
                exhibit_rows=len(exhibits),
                exhibit_labels=dict(Counter(r["exhibit_label"] for r in exhibits)),
                exhibit_types_seen=dict(types_seen),
                throttle_wait_sec=client.throttle_wait_sec,
            )

    print(
        f"Wrote {out_path}: {len(rows)} documents "
        f"({len(primaries)} filings + {len(exhibits)} exhibits), "
        f"{date_from} .. {date_to}"
    )
    for ticker, count in sorted(per_ticker.items()):
        print(f"  {ticker:<6} {count:>4}")
    by_form = Counter(r["form"] for r in primaries)
    print("  by form: " + ", ".join(f"{f}={n}" for f, n in sorted(by_form.items())))
    if indexed or index_failed:
        print(
            f"  exhibit index: {indexed} filing(s) indexed, "
            f"{with_exhibit} with a qualifying exhibit, "
            f"{without_exhibit} without, {index_failed} failed"
        )
        if exhibits:
            labels = Counter(r["exhibit_label"] for r in exhibits)
            print("  exhibit labels: "
                  + ", ".join(f"{k}={v}" for k, v in sorted(labels.items())))
    if shards_fetched:
        print(f"  fetched {shards_fetched} older filings shard(s) to cover the window")
    if truncated_ciks:
        print(f"  WARNING: manifest truncated for {', '.join(truncated_ciks)}")
    if failed:
        print(f"ERROR: {len(failed)} CIK(s) failed: {', '.join(failed)}", file=sys.stderr)
        return 1
    if index_failed:
        # Non-zero for the same reason a failed CIK is: the manifest is short by
        # an unknown number of exhibits, and a short manifest must never be
        # mistaken for a complete one.
        print(
            f"ERROR: {index_failed} filing index request(s) failed; their exhibits "
            f"are not in the manifest.",
            file=sys.stderr,
        )
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tools.discover",
        description="Stage 1: list 10-Q and 8-K filings in a filing-date window.",
    )
    add_config_args(parser)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the planned requests and window; make no request",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="only process the first N CIKs of the universe",
    )
    args = parser.parse_args(argv)

    try:
        return discover(
            args.config, args.overrides, dry_run=args.dry_run, limit=args.limit
        )
    except (ConfigError, EdgarError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
