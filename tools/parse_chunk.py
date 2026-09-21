"""Stage 3 -- parse+chunk: temp/filings/ -> temp/chunks/.

    python -m tools.parse_chunk
    python -m tools.parse_chunk --dry-run
    python -m tools.parse_chunk --limit 5
    python -m tools.parse_chunk --force
    python -m tools.parse_chunk --set chunk.size=300 --set chunk.overlap=60

Strips each fetched document to text, splits it on the configured section
anchors, then splits any oversized section on the token budget. Every chunk
carries full provenance.

Reads and writes local files only -- no network, so the run log shows
``requests: 0``.

WHAT THE SECTION DISTRIBUTION IS FOR
====================================

The stage reports chunk counts by section label, per form and document type,
before anything is embedded. Three buckets are kept distinct because they mean
different things:

  a named label   an anchor matched
  _default        the form had no anchor list of its own and the fallback matched
  (unsectioned)   NO anchor matched

The third is the health signal. Anchors are regexes over text extracted from
markup we do not control, and when they stop matching nothing fails: chunks
still get produced, the index still builds, and retrieval merely gets worse.
Measured against the corpus this was built on, the expected unsectioned char
share is roughly 2% for 10-Q, 42% for 8-K primaries (their cover page precedes
the first Item, which is not a failure), and ~0% for exhibits once labelled.
A 10-Q share far above that means the anchors need attention.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tools import ChunkProvenance
from tools.config import Config, ConfigError, add_config_args, load_config
from tools.edgar import (
    DOC_TYPE_EXHIBIT,
    DOC_TYPE_PRIMARY,
    EdgarError,
    as_stored_path,
    from_stored_path,
    stored_filename,
)
from tools.htmltext import extract_text, inline_tags_from_config
from tools.runlog import RunLog, StageRecorder
from tools.tokenize import TokenizerError, Windower, get_windower

__all__ = ["parse_chunk", "split_sections", "config_fingerprint"]

#: temp/chunks/parsed.csv -- stage 3 ledger, same role as the fetch ledger.
PARSE_LEDGER_FIELDS: tuple[str, ...] = (
    "accession",
    "document",
    "doc_sha256",
    "config_fingerprint",
    "chunks",
    "path",
    "parsed_at",
)

#: Printed and recorded for the bucket where no anchor matched. Not a label that
#: can ever come from config, so it cannot collide with a real section name.
UNSECTIONED = "(unsectioned)"


# --------------------------------------------------------------------------
# Config fingerprint
# --------------------------------------------------------------------------


def config_fingerprint(config: Config, windower: Windower) -> str:
    """Hash of every setting that changes chunk output.

    A document is re-parsed when this changes and not otherwise. The windower's
    identity is included because it names the tokenizer, and a token count from
    one tokenizer is not comparable with a count from another -- chunks would
    silently change size while every recorded number stayed the same.
    """
    material = {
        "parse": {
            key: config.get(f"parse.{key}", None)
            for key in (
                "section_anchors",
                "min_section_chars",
                "preamble_label",
                "anchor_flags",
                "inline_tags_no_separator",
                "exhibit_section_from_label",
            )
        },
        "chunk": {
            key: config.get(f"chunk.{key}", None)
            for key in ("unit", "size", "overlap", "snap_to_sentence", "snap_search_fraction")
        },
        "windower": windower.identity,
    }
    blob = json.dumps(material, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------------
# Section splitting
# --------------------------------------------------------------------------


def _anchors_for(config: Config, form: str, doc_type: str) -> list[dict[str, str]]:
    """Resolve the anchor list: '<form>/<doc_type>', then '<form>', then _default."""
    anchors = config.get("parse.section_anchors", {}) or {}
    for key in (f"{form}/{doc_type}", form, "_default"):
        found = anchors.get(key)
        if found:
            return list(found)
    return []


def _anchor_flags(config: Config) -> int:
    flags = 0
    for name in config.get("parse.anchor_flags", []) or []:
        flag = getattr(re, str(name).strip().upper(), None)
        if not isinstance(flag, re.RegexFlag):
            raise ConfigError(f"unknown parse.anchor_flags entry {name!r}")
        flags |= int(flag)
    return flags


def split_sections(
    text: str,
    anchors: list[dict[str, str]],
    *,
    min_section_chars: int,
    preamble_label: str | None,
    flags: int = re.IGNORECASE,
) -> list[tuple[str | None, int, int]]:
    """Split text into ``(label, start, end)`` spans using ordered anchors.

    Every anchor is matched everywhere, the matches are ordered by position, and
    each opens a section running to the next. A section shorter than
    ``min_section_chars`` is dropped as a table-of-contents false positive --
    necessary rather than tidy: on this corpus a single risk-factors anchor hits
    161 times across 39 documents, because a 10-Q lists its items in a contents
    table before the body. TOC entries sit close together and so fall under the
    threshold; the real heading opens a section that does not.

    A label containing ``{0}`` takes the match's first capture group, which is
    how one 8-K anchor covers every ``Item N.NN`` without enumerating them.

    Text before the first surviving anchor takes ``preamble_label`` -- null by
    config, because it is genuinely unattributed and a guessed label would
    destroy that signal (Rule 4).
    """
    if not text:
        return []

    hits: list[tuple[int, str]] = []
    for anchor in anchors:
        label_template = str(anchor["label"])
        for match in re.finditer(str(anchor["pattern"]), text, flags):
            label = label_template
            if "{0}" in label_template:
                if not match.groups() or not match.group(1):
                    continue
                label = label_template.format(match.group(1).strip())
            hits.append((match.start(), label))

    if not hits:
        return [(preamble_label, 0, len(text))]

    hits.sort(key=lambda h: h[0])

    # Reject contents-table entries by looking at BOTH neighbouring gaps.
    #
    # Dropping only short-next-gap hits is not enough, and the way it fails is
    # the dangerous kind. A contents table lists every item within a page or so,
    # so each entry has a short gap to the next -- except the LAST one, whose
    # next anchor is the first real heading tens of thousands of characters
    # away. That entry therefore survives and absorbs the entire body between
    # the contents table and the first matching heading. Observed on this
    # corpus: a 10-Q's financial statements, 54,352 characters, labelled
    # "part2_item6_exhibits" because the last contents entry was Item 6. The
    # chunks were well-formed and their section label was simply wrong, which no
    # downstream stage could detect.
    #
    # A contents entry is one of a CLUSTER, so it is close to a neighbour on one
    # side or the other. A real heading has room on both sides. Gaps are
    # measured against raw neighbours, not surviving ones, because it is the
    # crowding that identifies the cluster.
    kept_hits: list[tuple[int, str]] = []
    for i, (start, label) in enumerate(hits):
        next_pos = hits[i + 1][0] if i + 1 < len(hits) else len(text)
        if next_pos - start < min_section_chars:
            continue
        if i > 0 and start - hits[i - 1][0] < min_section_chars:
            continue
        kept_hits.append((start, label))

    if not kept_hits:
        return [(preamble_label, 0, len(text))]

    spans: list[tuple[str | None, int, int]] = []
    if kept_hits[0][0] > 0:
        spans.append((preamble_label, 0, kept_hits[0][0]))
    for i, (start, label) in enumerate(kept_hits):
        end = kept_hits[i + 1][0] if i + 1 < len(kept_hits) else len(text)
        spans.append((label, start, end))
    return spans


# --------------------------------------------------------------------------
# Ledger
# --------------------------------------------------------------------------


def _load_parse_ledger(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        header = tuple(reader.fieldnames or ())
        if header != PARSE_LEDGER_FIELDS:
            raise EdgarError(
                f"{path} has header {header!r}, expected {PARSE_LEDGER_FIELDS!r}. "
                f"Delete it to re-parse."
            )
        return {(r["accession"], r["document"]): dict(r) for r in reader}


def _write_parse_ledger(path: Path, rows: dict[tuple[str, str], dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=list(PARSE_LEDGER_FIELDS), lineterminator="\n"
        )
        writer.writeheader()
        for row in rows.values():
            writer.writerow({k: row.get(k, "") for k in PARSE_LEDGER_FIELDS})
    tmp.replace(path)


def _is_parsed(
    entry: dict[str, str] | None, doc_sha256: str, fingerprint: str
) -> bool:
    """Has this exact document already been parsed under this exact config?

    Both halves matter. The sha means a re-fetched document is re-parsed; the
    fingerprint means any change to an anchor, a threshold, the chunk size or
    the tokenizer re-parses everything. Neither alone is sufficient.
    """
    if entry is None:
        return False
    if entry.get("doc_sha256") != doc_sha256 or entry.get("config_fingerprint") != fingerprint:
        return False
    out = from_stored_path(entry.get("path", ""))
    return out.is_file()


# --------------------------------------------------------------------------
# Chunking one document
# --------------------------------------------------------------------------


def _chunk_document(
    row: dict[str, str],
    text: str,
    config: Config,
    windower: Windower,
    stage: StageRecorder,
) -> list[dict[str, Any]]:
    """Produce the chunk records for one document."""
    doc_type = row.get("doc_type") or DOC_TYPE_PRIMARY
    min_section_chars = int(config.get("parse.min_section_chars", 0))
    preamble_label = config.get("parse.preamble_label", None)
    exhibit_from_label = bool(config.get("parse.exhibit_section_from_label", True))

    if doc_type == DOC_TYPE_EXHIBIT and exhibit_from_label:
        # An exhibit is a press release with no Item structure; its section is
        # its EDGAR document type, taken from the manifest. Sourced, not
        # invented -- and it keeps the unsectioned bucket meaningful as a
        # signal about anchors rather than a bucket everything falls into.
        label = (row.get("exhibit_label") or "").strip() or None
        spans: list[tuple[str | None, int, int]] = [(label, 0, len(text))]
    else:
        spans = split_sections(
            text,
            _anchors_for(config, row["form"], doc_type),
            min_section_chars=min_section_chars,
            preamble_label=preamble_label,
            flags=_anchor_flags(config),
        )

    records: list[dict[str, Any]] = []
    ordinal = 0
    for label, sec_start, sec_end in spans:
        section_text = text[sec_start:sec_end]
        if not section_text.strip():
            continue
        for win_start, win_end, n_units in windower.windows(section_text):
            start = sec_start + win_start
            end = sec_start + win_end
            body = text[start:end].strip()
            if not body:
                continue
            # Constructed through the frozen dataclass so a field that went
            # missing upstream fails here rather than reaching the index.
            provenance = ChunkProvenance(
                cik=row["cik"],
                ticker=row.get("ticker") or None,
                accession=row["accession"],
                form=row["form"],
                filing_date=row.get("filing_date") or None,
                period=row.get("period_of_report") or None,
                section=label,
                chunk_ordinal=ordinal,
            )
            missing = provenance.missing_fields()
            if missing:
                stage.error(
                    "chunk provenance has null field(s); kept and recorded",
                    context={
                        "accession": row["accession"],
                        "document": row.get("document"),
                        "chunk_ordinal": ordinal,
                        "missing": missing,
                    },
                )
            records.append(
                {
                    **provenance.to_dict(),
                    "chunk_id": f"{row['accession']}:{row.get('document') or ''}:{ordinal:05d}",
                    "text": body,
                    "n_tokens": n_units,
                    "char_span": [start, end],
                    # Beyond ChunkProvenance, which predates exhibits and has no
                    # field naming which document in a filing a chunk came from.
                    "doc_type": doc_type,
                    "document": row.get("document") or None,
                    "exhibit_label": row.get("exhibit_label") or None,
                }
            )
            ordinal += 1
    return records


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def _distribution(records_by_group: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Chunk and character counts by section label, per form/doc_type group."""
    report: dict[str, Any] = {}
    for group, records in sorted(records_by_group.items()):
        labels: Counter[str] = Counter()
        chars: Counter[str] = Counter()
        for rec in records:
            key = rec["section"] if rec["section"] is not None else UNSECTIONED
            labels[key] += 1
            chars[key] += len(rec["text"])
        total_chunks = sum(labels.values())
        total_chars = sum(chars.values())
        report[group] = {
            "chunks": total_chunks,
            "chars": total_chars,
            "unsectioned_chunks": labels.get(UNSECTIONED, 0),
            "unsectioned_char_share": (
                round(100.0 * chars.get(UNSECTIONED, 0) / total_chars, 2)
                if total_chars
                else 0.0
            ),
            "by_label": {
                label: {
                    "chunks": n,
                    "chars": chars[label],
                    "pct_chunks": round(100.0 * n / total_chunks, 2) if total_chunks else 0.0,
                    "pct_chars": round(100.0 * chars[label] / total_chars, 2) if total_chars else 0.0,
                }
                for label, n in labels.most_common()
            },
        }
    return report


def _print_distribution(report: dict[str, Any]) -> None:
    print("\n=== CHUNKS BY SECTION LABEL, PER FORM/DOC TYPE ===")
    for group, data in report.items():
        print(
            f"\n  {group}   {data['chunks']:,} chunks, {data['chars']:,} chars"
            f"   unsectioned: {data['unsectioned_char_share']}% of chars"
        )
        print(f"    {'section label':<36} {'chunks':>8} {'% chunks':>9} {'% chars':>8}")
        for label, d in data["by_label"].items():
            mark = "  <-- no anchor matched" if label == UNSECTIONED else ""
            print(
                f"    {label:<36} {d['chunks']:>8,} {d['pct_chunks']:>8.1f}% "
                f"{d['pct_chars']:>7.1f}%{mark}"
            )


# --------------------------------------------------------------------------
# Stage
# --------------------------------------------------------------------------


def parse_chunk(
    config_path: str,
    overrides: list[str],
    *,
    dry_run: bool = False,
    limit: int | None = None,
    force: bool = False,
) -> int:
    from tools.edgar import read_manifest

    config = load_config(config_path, overrides=overrides)

    manifest_path = Path(config.get("discover.out_path"))
    fetch_ledger_path = Path(config.get("fetch.ledger_path"))
    out_dir = Path(config.get("chunk.out_dir"))
    ledger_path = Path(config.get("chunk.ledger_path"))

    manifest = read_manifest(manifest_path)
    if limit is not None:
        manifest = manifest[:limit]

    with fetch_ledger_path.open("r", encoding="utf-8", newline="") as fh:
        fetched = {
            (r["accession"], r.get("document", "") or ""): r
            for r in csv.DictReader(fh)
        }

    windower = get_windower(config)
    fingerprint = config_fingerprint(config, windower)
    ledger = {} if force else _load_parse_ledger(ledger_path)
    inline_tags = inline_tags_from_config(config)

    pending: list[tuple[dict[str, str], dict[str, str]]] = []
    skipped = 0
    missing = 0
    for row in manifest:
        entry = fetched.get((row["accession"], row.get("document", "") or ""))
        if not entry or entry.get("status") != "ok":
            missing += 1
            continue
        if _is_parsed(
            ledger.get((row["accession"], row.get("document", "") or "")),
            entry.get("sha256", ""),
            fingerprint,
        ):
            skipped += 1
            continue
        pending.append((row, entry))

    if dry_run:
        print("DRY RUN -- nothing written.")
        print(f"  manifest        {manifest_path} ({len(manifest)} rows)")
        print(f"  fetch ledger    {fetch_ledger_path} ({len(fetched)} entries)")
        print(f"  parse ledger    {ledger_path} ({len(ledger)} entries)")
        print(f"  fingerprint     {fingerprint}")
        print(f"  unit            {config.get('chunk.unit')} "
              f"size={config.get('chunk.size')} overlap={config.get('chunk.overlap')}")
        print(f"  windower        {windower.identity}")
        print(f"  would skip      {skipped} already parsed under this config")
        print(f"  would parse     {len(pending)} document(s)")
        print(f"  not fetched     {missing}")
        print(f"  would write     {out_dir}/<accession>[-<document>].jsonl")
        return 0

    failed = 0
    total_chunks = 0
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)

    with RunLog(config) as log:
        with log.stage("parse_chunk") as stage:
            stage.note(
                manifest_rows=len(manifest),
                config_fingerprint=fingerprint,
                windower=windower.identity,
                unit=config.get("chunk.unit"),
                chunk_size=config.get("chunk.size"),
                overlap=config.get("chunk.overlap"),
                snap_to_sentence=config.get("chunk.snap_to_sentence"),
                forced=force,
            )
            out_dir.mkdir(parents=True, exist_ok=True)
            if missing:
                stage.error(
                    "manifest rows have no successfully fetched document; skipped",
                    context={"count": missing},
                )

            for row, entry in pending:
                doc_type = row.get("doc_type") or DOC_TYPE_PRIMARY
                try:
                    raw = from_stored_path(entry["path"]).read_bytes()
                    stage.bytes_in(len(raw))
                    text = extract_text(raw, inline_tags=inline_tags)
                    records = _chunk_document(row, text, config, windower, stage)
                    if not records:
                        stage.error(
                            "document produced no chunks",
                            context={
                                "accession": row["accession"],
                                "document": row.get("document"),
                                "chars": len(text),
                            },
                        )
                    out_path = out_dir / stored_filename(
                        row["accession"], doc_type, row.get("document") or "", ".jsonl"
                    )
                    payload = "".join(
                        json.dumps(r, ensure_ascii=False) + "\n" for r in records
                    )
                    tmp = out_path.with_name(out_path.name + ".part")
                    tmp.write_text(payload, encoding="utf-8")
                    tmp.replace(out_path)

                    stage.count(len(records))
                    stage.bytes_out(len(payload.encode("utf-8")))
                    total_chunks += len(records)
                    by_group[f"{row['form']}/{doc_type}"].extend(records)
                    ledger[(row["accession"], row.get("document", "") or "")] = {
                        "accession": row["accession"],
                        "document": row.get("document") or "",
                        "doc_sha256": entry.get("sha256", ""),
                        "config_fingerprint": fingerprint,
                        "chunks": len(records),
                        "path": as_stored_path(out_path),
                        "parsed_at": datetime.now(timezone.utc).isoformat(
                            timespec="seconds"
                        ),
                    }
                except (OSError, TokenizerError, ValueError, TypeError) as exc:
                    failed += 1
                    stage.error(
                        exc,
                        context={
                            "accession": row["accession"],
                            "document": row.get("document"),
                        },
                    )

            _write_parse_ledger(ledger_path, ledger)
            distribution = _distribution(by_group)
            stage.note(
                documents=len(manifest),
                parsed=len(pending) - failed,
                skipped=skipped,
                failed=failed,
                not_fetched=missing,
                chunks=total_chunks,
                chunks_by_section=distribution,
                unsectioned_char_share={
                    g: d["unsectioned_char_share"] for g, d in distribution.items()
                },
                out_dir=str(out_dir),
                ledger_path=str(ledger_path),
            )

    print(f"Parsed {len(pending) - failed} document(s) into {out_dir}")
    print(f"  {total_chunks:,} chunks | skipped {skipped} already parsed | "
          f"{failed} failed | {missing} not fetched")
    print(f"  fingerprint {fingerprint} | {windower.identity} "
          f"size={config.get('chunk.size')} overlap={config.get('chunk.overlap')}")
    if distribution:
        _print_distribution(distribution)
    if failed:
        print(f"\nERROR: {failed} document(s) failed to parse.", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tools.parse_chunk",
        description="Stage 3: strip, section-split and chunk the fetched corpus.",
    )
    add_config_args(parser)
    parser.add_argument(
        "--dry-run", action="store_true", help="report what would be parsed; write nothing"
    )
    parser.add_argument(
        "--limit", type=int, default=None, metavar="N",
        help="only process the first N manifest rows",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="ignore the parse ledger and re-parse every document",
    )
    args = parser.parse_args(argv)

    try:
        return parse_chunk(
            args.config, args.overrides,
            dry_run=args.dry_run, limit=args.limit, force=args.force,
        )
    except (ConfigError, EdgarError, TokenizerError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
