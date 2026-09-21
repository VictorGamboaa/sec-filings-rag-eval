"""Structural audit of the fetched corpus.

    python -m tools.audit_corpus
    python -m tools.audit_corpus --json-only
    python -m tools.audit_corpus --set audit.short_item_threshold=1000

Answers one question: is what is on disk structurally capable of supporting
retrieval, or are there documents that are pointers to content we do not have?

Reads local files only -- the manifest, the fetch ledger, and the fetched
documents. It makes no network request, so its run log correctly shows
``requests: 0``.

WHAT IT MEASURES, AND WHY THESE MEASURES
========================================

*Body text volume* alone does not answer the question. An 8-K whose entire
narrative is "see the press release attached as Exhibit 99.1" still carries
2,000-3,000 characters of mandatory cover page -- registrant, address, checkbox
paragraphs, signature -- so a naive character threshold passes every document and
detects nothing.

So the audit also splits body text at the first item anchor and again at the
signature block. The span between them is the filing's own narrative, as
distinct from the cover page around it and from content it incorporates by
reference. A short item narrative next to an exhibit reference is the shape of a
pointer.

*Exhibit resolution* is the direct measure: for every EX-99.x a document
references, is that exhibit present on disk? Before exhibits were fetched this
was zero by construction. After, it is the number that says whether the gap
closed.

RULE 6
======

This tool reports structure and counts. It does not characterize what any filing
says, and it emits no conclusion or recommendation. "This document is 3,604
characters and references an exhibit that is not on disk" is a fact about files.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from tools.config import Config, ConfigError, add_config_args, load_config
from tools.edgar import (
    DOC_TYPE_EXHIBIT,
    DOC_TYPE_PRIMARY,
    EdgarError,
    from_stored_path,
    read_manifest,
)
from tools.htmltext import extract_text, inline_tags_from_config
from tools.runlog import RunLog

__all__ = ["audit_corpus"]


def _load_ledger_paths(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    """Read the fetch ledger keyed by (accession, document), tolerating legacy.

    The audit must work against a corpus fetched before exhibits existed, since
    that is exactly the state it is used to diagnose. A legacy ledger has no
    ``document`` column, so its rows land under an empty document name; see
    ``_ledger_entry`` for how those are matched.
    """
    import csv

    if not path.is_file():
        raise EdgarError(
            f"{path} not found. Fetch documents first: python -m tools.fetch"
        )
    with path.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    return {(r["accession"], r.get("document", "") or ""): r for r in rows}


def _ledger_entry(
    ledger: dict[tuple[str, str], dict[str, str]], row: dict[str, str]
) -> dict[str, str] | None:
    """Find a manifest row's ledger entry, across both ledger schemas.

    Exact ``(accession, document)`` first. Failing that, a PRIMARY row also
    matches an entry filed under an empty document name, which is what a legacy
    ledger produces -- those rows are primaries by construction. An exhibit row
    never falls back, because an empty-named entry can only be a primary.
    """
    entry = ledger.get((row["accession"], row.get("document", "") or ""))
    if entry is not None:
        return entry
    if (row.get("doc_type") or DOC_TYPE_PRIMARY) == DOC_TYPE_PRIMARY:
        return ledger.get((row["accession"], ""))
    return None


def _split_item_narrative(
    text: str, item_re: re.Pattern[str], sig_re: re.Pattern[str]
) -> tuple[int, list[str]]:
    """Characters between the first item anchor and the signature block."""
    first = item_re.search(text)
    if first is None:
        return 0, []
    sigs = list(sig_re.finditer(text, first.start()))
    end = sigs[-1].start() if sigs else len(text)
    return len(text[first.start() : end].strip()), sorted(set(item_re.findall(text)))


def _is_resolved(label: str, present: set[str]) -> bool:
    """Is a referenced exhibit label satisfied by what the filing has on disk?

    Exact match first. A label with no sub-number ("99", from prose reading
    "Exhibit 99") is satisfied by any numbered exhibit of that series, because
    "99" and "99.1" name the same document when the filing carries only one --
    requiring an exact string match there would report a gap that does not
    exist.
    """
    if label in present:
        return True
    if "." not in label:
        return any(p == label or p.startswith(f"{label}.") for p in present)
    return False


def _percentiles(values: list[int]) -> dict[str, int | None]:
    if not values:
        return {"min": None, "median": None, "max": None}
    ordered = sorted(values)
    return {
        "min": ordered[0],
        "median": int(statistics.median(ordered)),
        "max": ordered[-1],
    }


def audit_corpus(config: Config, *, json_only: bool = False) -> dict[str, Any]:
    """Score every fetched document. Returns the report as a dict."""
    manifest_path = Path(config.get("discover.out_path"))
    ledger_path = Path(config.get("fetch.ledger_path"))

    item_re = re.compile(config.get("audit.item_pattern"), re.I)
    sig_re = re.compile(config.get("audit.signature_pattern"), re.I)
    exhibit_re = re.compile(config.get("audit.exhibit_pattern"), re.I)
    inline_tags = inline_tags_from_config(config)
    short_text = int(config.get("audit.short_text_threshold"))
    short_item = int(config.get("audit.short_item_threshold"))

    # allow_legacy: the audit is the tool you reach for to describe a corpus
    # that predates the exhibit schema, so it must be able to read one.
    manifest = read_manifest(manifest_path, allow_legacy=True)
    ledger = _load_ledger_paths(ledger_path)

    # Which exhibit labels does each filing actually have on disk?
    exhibits_on_disk: dict[str, set[str]] = defaultdict(set)
    for row in manifest:
        if row.get("doc_type") != DOC_TYPE_EXHIBIT:
            continue
        entry = _ledger_entry(ledger, row)
        if entry and entry.get("status") == "ok" and from_stored_path(entry["path"]).is_file():
            label = (row.get("exhibit_label") or "").upper().replace("EX-", "")
            exhibits_on_disk[row["accession"]].add(label)

    documents: list[dict[str, Any]] = []
    missing_on_disk = 0

    for row in manifest:
        entry = _ledger_entry(ledger, row)
        if not entry or entry.get("status") != "ok":
            missing_on_disk += 1
            continue
        path = from_stored_path(entry["path"])
        if not path.is_file():
            missing_on_disk += 1
            continue

        text = extract_text(path.read_bytes(), inline_tags=inline_tags)
        item_chars, items = _split_item_narrative(text, item_re, sig_re)
        referenced = sorted(set(exhibit_re.findall(text)))
        present = exhibits_on_disk.get(row["accession"], set())
        unresolved = [
            label for label in referenced if not _is_resolved(label, present)
        ]

        documents.append(
            {
                "ticker": row["ticker"],
                "accession": row["accession"],
                "filing_date": row["filing_date"],
                "form": row["form"],
                "doc_type": row.get("doc_type") or DOC_TYPE_PRIMARY,
                "exhibit_label": row.get("exhibit_label") or None,
                "document": row.get("document") or None,
                "file": path.as_posix(),
                "bytes": int(entry.get("bytes") or 0),
                "body_chars": len(text),
                "item_chars": item_chars,
                "cover_chars": len(text) - item_chars,
                "items": items,
                "exhibits_referenced": referenced,
                "exhibits_unresolved": unresolved,
            }
        )

    report = _summarize(
        documents,
        manifest=manifest,
        missing_on_disk=missing_on_disk,
        short_text=short_text,
        short_item=short_item,
        config=config,
    )
    report["documents"] = documents
    if not json_only:
        _print_report(report, config)
    return report


def _summarize(
    documents: list[dict[str, Any]],
    *,
    manifest: list[dict[str, str]],
    missing_on_disk: int,
    short_text: int,
    short_item: int,
    config: Config,
) -> dict[str, Any]:
    by_form: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for doc in documents:
        by_form[f"{doc['form']}/{doc['doc_type']}"].append(doc)

    primaries_8k = [
        d for d in documents if d["form"] == "8-K" and d["doc_type"] == DOC_TYPE_PRIMARY
    ]
    referencing = [d for d in primaries_8k if d["exhibits_referenced"]]
    unresolved = [d for d in primaries_8k if d["exhibits_unresolved"]]
    item_202 = [d for d in primaries_8k if "2.02" in d["items"]]
    item_202_unresolved = [d for d in item_202 if d["exhibits_unresolved"]]

    groups = {}
    for key, docs in sorted(by_form.items()):
        body = [d["body_chars"] for d in docs]
        # The item split is only meaningful where the configured item pattern
        # actually matched. It is written for the 8-K "N.NN" form, so it finds
        # nothing in a 10-Q -- and reporting a median of 0 there would read as
        # "these filings have no narrative" rather than "this measure does not
        # apply to them". Documents with no match are excluded and counted.
        matched = [d["item_chars"] for d in docs if d["item_chars"] > 0]
        groups[key] = {
            "count": len(docs),
            "body_chars": _percentiles(body),
            "item_anchor_matched": len(matched),
            "item_chars": _percentiles(matched),
            "under_short_text": sum(1 for v in body if v < short_text),
            "under_short_item": sum(1 for v in matched if v < short_item),
            "total_body_chars": sum(body),
        }

    per_ticker = {}
    for ticker in sorted({d["ticker"] for d in documents}):
        docs = [d for d in documents if d["ticker"] == ticker]
        prim = [d for d in docs if d["doc_type"] == DOC_TYPE_PRIMARY]
        exh = [d for d in docs if d["doc_type"] == DOC_TYPE_EXHIBIT]
        per_ticker[ticker] = {
            "documents": len(docs),
            "primary": len(prim),
            "exhibit": len(exh),
            "body_chars": _percentiles([d["body_chars"] for d in docs]),
            "unresolved_exhibit_refs": sum(1 for d in docs if d["exhibits_unresolved"]),
        }

    return {
        "thresholds": {
            "short_text_threshold": short_text,
            "short_item_threshold": short_item,
        },
        "manifest_rows": len(manifest),
        "documents_scored": len(documents),
        "documents_missing_on_disk": missing_on_disk,
        "by_form": groups,
        "by_ticker": per_ticker,
        "exhibit_resolution": {
            "8k_primary_documents": len(primaries_8k),
            "referencing_an_exhibit": len(referencing),
            "with_unresolved_reference": len(unresolved),
            "resolved": len(referencing) - len(unresolved),
            "item_2_02_documents": len(item_202),
            "item_2_02_with_unresolved_reference": len(item_202_unresolved),
        },
        "total_body_chars": sum(d["body_chars"] for d in documents),
    }


def _fmt(value: Any) -> str:
    return "n/a" if value is None else f"{value:,}"


def _print_report(report: dict[str, Any], config: Config) -> None:
    print("=== CORPUS AUDIT (structure only) ===")
    print(f"  manifest rows      : {report['manifest_rows']:,}")
    print(f"  documents scored   : {report['documents_scored']:,}")
    print(f"  missing on disk    : {report['documents_missing_on_disk']:,}")
    print(f"  total body chars   : {report['total_body_chars']:,}")
    print()

    print("=== BY FORM / DOCUMENT TYPE ===")
    print(f"  {'group':<18} {'n':>5} {'body min':>10} {'body med':>10} "
          f"{'body max':>10} {'items':>6} {'item med':>9} {'short':>6}")
    for key, g in report["by_form"].items():
        b, i = g["body_chars"], g["item_chars"]
        print(f"  {key:<18} {g['count']:>5} {_fmt(b['min']):>10} "
              f"{_fmt(b['median']):>10} {_fmt(b['max']):>10} "
              f"{g['item_anchor_matched']:>6} {_fmt(i['median']):>9} "
              f"{g['under_short_text']:>6}")
    print(f"  ('short' = body text under {report['thresholds']['short_text_threshold']:,} "
          f"chars; 'items' = documents where the item pattern matched, and the")
    print("   item median is taken over those only -- the pattern is the 8-K "
          "N.NN form and does not apply to 10-Q)")
    print()

    ex = report["exhibit_resolution"]
    print("=== EXHIBIT RESOLUTION (the pointer check) ===")
    print(f"  8-K primary documents            : {ex['8k_primary_documents']:,}")
    print(f"  referencing an Exhibit 99.x      : {ex['referencing_an_exhibit']:,}")
    print(f"  ...resolved (exhibit on disk)    : {ex['resolved']:,}")
    print(f"  ...UNRESOLVED (not on disk)      : {ex['with_unresolved_reference']:,}")
    print(f"  Item 2.02 documents              : {ex['item_2_02_documents']:,}")
    print(f"  ...with an unresolved reference  : {ex['item_2_02_with_unresolved_reference']:,}")
    print()

    print("=== BY TICKER ===")
    print(f"  {'ticker':<8} {'docs':>6} {'primary':>8} {'exhibit':>8} "
          f"{'body med':>10} {'unresolved':>11}")
    for ticker, t in report["by_ticker"].items():
        print(f"  {ticker:<8} {t['documents']:>6} {t['primary']:>8} "
              f"{t['exhibit']:>8} {_fmt(t['body_chars']['median']):>10} "
              f"{t['unresolved_exhibit_refs']:>11}")
    print()

    _print_sample(report, config)


def _print_sample(report: dict[str, Any], config: Config) -> None:
    """A reproducible per-ticker sample, for eyeballing actual numbers."""
    import random

    per_ticker = int(config.get("audit.sample_per_ticker"))
    seed = config.get("audit.sample_seed")
    rng = random.Random(seed)

    pools: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for doc in report["documents"]:
        if doc["form"] == "8-K" and doc["doc_type"] == DOC_TYPE_PRIMARY:
            pools[doc["ticker"]].append(doc)

    print(f"=== SAMPLE ({per_ticker} 8-K primaries per ticker, seed {seed}) ===")
    print(f"  {'ticker':<7} {'filed':<12} {'body':>7} {'cover':>7} {'item':>7}  "
          f"{'items':<16} {'refs':<8} unresolved")
    for ticker in sorted(pools):
        pool = sorted(pools[ticker], key=lambda d: d["accession"])
        for doc in rng.sample(pool, min(per_ticker, len(pool))):
            print(
                f"  {doc['ticker']:<7} {doc['filing_date']:<12} "
                f"{doc['body_chars']:>7,} {doc['cover_chars']:>7,} "
                f"{doc['item_chars']:>7,}  "
                f"{','.join(doc['items']) or '-':<16} "
                f"{','.join(doc['exhibits_referenced']) or '-':<8} "
                f"{','.join(doc['exhibits_unresolved']) or 'none'}"
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tools.audit_corpus",
        description="Structural audit of the fetched corpus. Reads local files only.",
    )
    add_config_args(parser)
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="write the JSON report without printing the human-readable one",
    )
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config, overrides=args.overrides)
        out_dir = Path(config.get("audit.out_dir"))

        with RunLog(config) as log:
            with log.stage("audit_corpus") as stage:
                report = audit_corpus(config, json_only=args.json_only)
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"{log.run_id}.json"
                payload = json.dumps(report, indent=2, default=str)
                out_path.write_text(payload, encoding="utf-8")

                stage.count(report["documents_scored"])
                stage.bytes_in(report["total_body_chars"])
                stage.bytes_out(len(payload.encode("utf-8")))
                # No request() calls: this stage reads local files only, and the
                # zero in the run log is what says so.
                stage.note(
                    out_path=str(out_path),
                    documents_scored=report["documents_scored"],
                    documents_missing_on_disk=report["documents_missing_on_disk"],
                    **{f"exhibit_{k}": v for k, v in report["exhibit_resolution"].items()},
                )
                if report["documents_missing_on_disk"]:
                    stage.error(
                        "manifest rows have no fetched document on disk",
                        context={"count": report["documents_missing_on_disk"]},
                    )
        print(f"\nReport written to {out_path}")
        return 0
    except (ConfigError, EdgarError, re.error) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
