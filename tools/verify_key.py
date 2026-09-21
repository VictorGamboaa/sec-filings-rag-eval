"""Verify answer-key quotes against the built index.

    python -m tools.verify_key
    python -m tools.verify_key --json-only
    python -m tools.verify_key --set evaluate.generic_match_threshold=5
    python -m tools.verify_key --only q01,q13

For every ground-truth entry carrying a quote, finds the chunks whose text
contains that quote AND whose accession and document match, and reports what it
found.

SCOPING: ACCESSION IS NOT ENOUGH
================================

An accession identifies a FILING, and a filing owns its primary document and its
exhibits alike. Scoping to the accession alone would let a match in an 8-K shell
satisfy a ground truth that lives in the attached press release -- erasing
exactly the distinction the exhibit expansion was built to make measurable. A
ground truth naming a ``document`` is therefore scoped to that document.

``accession: ANY`` opts out: the quote is searched across the whole corpus and
the report gives the number of DISTINCT accessions holding it. That is a
deduplication measurement, not a retrieval target -- a string in one filing is a
fact about that filing, the same string in thirty is boilerplate and recall over
it means nothing. Such an entry is never flagged "too generic"; breadth is the
finding, not a defect.

READ-ONLY, AND LOCAL-ONLY
========================

The key is never modified -- not a field, not a section label, not a dropped
entry. It is the record of what someone decided the right answer is, and a tool
that edited it would destroy the thing it is supposed to check. Everything here
is a report; acting on it is a human decision.

No network either. The run log shows ``requests: 0``.

WHAT A RESULT MEANS
===================

``0 matches`` is the important one, and it has several distinguishable causes.
The report separates them rather than leaving one number to be guessed at:

  accession not indexed   the filing is not in the corpus at all, so the quote
                          could not have been found regardless
  document not indexed    the filing is indexed, but not that document of it
  quote absent            the document is indexed but the quote appears nowhere
                          in its text -- the key's quote is paraphrased, not
                          verbatim
  split across chunks     the quote IS in the document text but crosses a chunk
                          boundary, so no single chunk contains all of it

Only "quote absent" means the key is wrong. "Split across chunks" means the key
is right and the chunking split it, which is a scoring problem, not a key one.

A ground truth with no quote reads as ``awaiting_quote`` when its entry declares
``status: NEEDS_QUOTE`` -- the author knows it is missing and is coming back for
it. Only an unexplained missing quote is ``no_quote``. Merging the two would
file planned work as a defect.

``> N matches`` (``evaluate.generic_match_threshold``) means the quote is too
generic to score cleanly -- it appears in enough chunks that a retrieval result
containing "the" match is not evidence of anything.

MATCHING
========

Substring, after collapsing runs of whitespace on both sides. The collapse is
not fuzziness: it is the same normalization ``tools.htmltext`` already applies
when extracting text, so comparing raw key text against collapsed chunk text
would fail on formatting alone. Nothing else is normalized -- no case folding,
no punctuation substitution, no approximate matching. A near-match reported as a
match would manufacture agreement that the key does not actually have.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from tools.config import Config, ConfigError, add_config_args, load_config
from tools.edgar import from_stored_path
from tools.htmltext import extract_text, inline_tags_from_config
from tools.runlog import RunLog
from tools.embed_index import METADATA_FILE

__all__ = ["verify_key", "normalize"]

_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Collapse whitespace. The only transformation applied to either side."""
    return _WS.sub(" ", str(text)).strip()


def _load_key(path: Path) -> list[dict[str, Any]]:
    import yaml

    if not path.is_file():
        raise ConfigError(
            f"{path} not found. Point evaluate.answer_key_path at the key, or "
            f"create it."
        )
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ConfigError(
            f"{path} must be a list of entries, got {type(data).__name__}."
        )
    return data


def _ground_truths(entry: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize the three shapes ``ground_truth`` takes in the key.

    A mapping is one ground truth; a list is several (cross-company and
    period-over-period questions need more than one); ``null`` is a negative
    control, which is *supposed* to have none and is not a defect.
    """
    gt = entry.get("ground_truth")
    if gt is None:
        return []
    if isinstance(gt, dict):
        return [gt]
    if isinstance(gt, list):
        return [g for g in gt if isinstance(g, dict)]
    return []


def _load_metadata(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ConfigError(
            f"{path} not found. Build the index first: python -m tools.embed_index"
        )
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _document_text(
    accession: str, document: str | None, ledger: dict[tuple[str, str], dict[str, str]],
    config: Config, cache: dict[tuple[str, str], str],
) -> str | None:
    """Full extracted text of a fetched document, for the split-vs-absent test."""
    key = (accession, document or "")
    if key in cache:
        return cache[key]
    entry = ledger.get(key)
    if entry is None and document:
        # The key names a document; fall back to any document of that accession
        # only if the named one is absent from the ledger.
        entry = next(
            (v for (acc, _), v in ledger.items() if acc == accession), None
        )
    if entry is None or entry.get("status") != "ok":
        cache[key] = None  # type: ignore[assignment]
        return None
    path = from_stored_path(entry["path"])
    if not path.is_file():
        cache[key] = None  # type: ignore[assignment]
        return None
    text = normalize(
        extract_text(path.read_bytes(), inline_tags=inline_tags_from_config(config))
    )
    cache[key] = text
    return text


def verify_key(
    config: Config, *, only: set[str] | None = None, json_only: bool = False
) -> dict[str, Any]:
    key_path = Path(config.get("evaluate.answer_key_path"))
    index_dir = Path(config.get("index.out_dir")) / str(config.get("index.name"))
    metadata_path = index_dir / METADATA_FILE
    ledger_path = Path(config.get("fetch.ledger_path"))
    threshold = int(config.get("evaluate.generic_match_threshold", 3))

    key = _load_key(key_path)
    metadata = _load_metadata(metadata_path)

    by_accession: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in metadata:
        by_accession[row["accession"]].append(row)
    # Normalize chunk text once, not once per quote.
    normalized: dict[int, str] = {
        id(row): normalize(row["text"]) for row in metadata
    }

    ledger: dict[tuple[str, str], dict[str, str]] = {}
    if ledger_path.is_file():
        import csv

        with ledger_path.open("r", encoding="utf-8", newline="") as fh:
            for r in csv.DictReader(fh):
                ledger[(r["accession"], r.get("document", "") or "")] = r
    doc_cache: dict[tuple[str, str], str] = {}

    results: list[dict[str, Any]] = []
    for entry in key:
        entry_id = str(entry.get("id"))
        if only and entry_id not in only:
            continue
        key_status = str(entry.get("status") or "").strip()
        truths = _ground_truths(entry)
        if not truths:
            # Two different things look identical here and must not be merged.
            # A negative control is SUPPOSED to have no ground truth -- that is
            # the question type. An unwritten entry of any other type is work
            # outstanding. Reporting both as "no ground truth" would say the
            # unwritten ones are fine by design.
            expected = str(entry.get("type")) == "negative_control"
            results.append(
                {
                    "id": entry_id,
                    "type": entry.get("type"),
                    "key_status": key_status,
                    "status": "negative_control" if expected else "ground_truth_unwritten",
                    "quotes": [],
                }
            )
            continue

        quote_results = []
        for position, truth in enumerate(truths):
            quote = truth.get("quote")
            accession = truth.get("accession")
            if not quote:
                # NEEDS_QUOTE says the author knows the quote is missing and is
                # coming back for it. Reporting that as a failure alongside a
                # quote that turned out to be paraphrased would merge a planned
                # state with a defect.
                quote_results.append(
                    {
                        "position": position,
                        "accession": accession,
                        "ticker": truth.get("ticker"),
                        "document": truth.get("document"),
                        "status": (
                            "awaiting_quote"
                            if key_status == "NEEDS_QUOTE"
                            else "no_quote"
                        ),
                        "match_count": None,
                        "matches": [],
                    }
                )
                continue

            needle = normalize(quote)
            document = truth.get("document")
            any_accession = str(accession).strip().upper() == "ANY"
            any_document = document is None or str(document).strip().upper() == "ANY"

            if any_accession:
                # A dedup diagnostic asks how widely a string appears, so it is
                # searched corpus-wide on purpose rather than scoped to a filing.
                candidates = metadata
            else:
                candidates = by_accession.get(str(accession), [])
                if candidates and not any_document:
                    # Scope to the named document. An accession covers a filing's
                    # primary AND its exhibits, so accession alone would let a
                    # match in the primary satisfy a ground truth that lives in
                    # the exhibit -- which is precisely the distinction the
                    # exhibit expansion exists to make measurable.
                    scoped = [r for r in candidates if r["document"] == document]
                    candidates = scoped

            matches = [
                {
                    "chunk_id": row["chunk_id"],
                    "accession": row["accession"],
                    "section": row["section"],
                    "doc_type": row["doc_type"],
                    "document": row["document"],
                    "n_tokens": row["n_tokens"],
                }
                for row in candidates
                if needle in normalized[id(row)]
            ]

            if matches and any_accession:
                # Never "too generic": breadth is the measurement, not a defect.
                status = "dedup_measured"
            elif matches:
                status = "ok" if len(matches) <= threshold else "too_generic"
            elif any_accession:
                status = "quote_absent"
            elif not by_accession.get(str(accession)):
                status = "accession_not_indexed"
            elif not candidates:
                status = "document_not_indexed"
            else:
                # Distinguish "the key's quote is not in the document" from
                # "it is, but no single chunk holds all of it".
                text = _document_text(
                    str(accession), truth.get("document"), ledger, config, doc_cache
                )
                if text is None:
                    status = "document_unavailable"
                elif needle in text:
                    status = "split_across_chunks"
                else:
                    status = "quote_absent"

            result = {
                "position": position,
                "accession": accession,
                "ticker": truth.get("ticker"),
                "document": truth.get("document"),
                "key_section": truth.get("section"),
                "quote": quote,
                "quote_chars": len(needle),
                "status": status,
                "match_count": len(matches),
                "matches": matches,
            }
            if any_accession:
                # The dedup measurement: how widely does this text occur? A
                # quote in one accession is a fact about a filing; the same
                # quote in thirty is boilerplate, and recall over it is
                # meaningless.
                result["distinct_accessions"] = len({m["accession"] for m in matches})
                result["distinct_texts"] = len(
                    {normalized[id(row)] for row in candidates
                     if needle in normalized[id(row)]}
                )
                result["distinct_tickers"] = sorted(
                    {row["ticker"] for row in candidates
                     if needle in normalized[id(row)]}
                )
            quote_results.append(result)

        results.append(
            {
                "id": entry_id,
                "type": entry.get("type"),
                "key_status": key_status,
                "status": "checked",
                "quotes": quote_results,
            }
        )

    report = {
        "answer_key": str(key_path),
        "index": str(index_dir),
        "entries": len(results),
        "generic_match_threshold": threshold,
        "summary": _summarize(results),
        "results": results,
    }
    if not json_only:
        _print_report(report)
    return report


def _summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = defaultdict(int)
    for entry in results:
        # The key's own declared status, reported alongside what verification
        # found. The two answer different questions: one is what the author
        # believes, the other is what the index says.
        counts[f"key_status_{entry.get('key_status') or 'unset'}"] += 1
    for entry in results:
        if entry["status"] == "negative_control":
            counts["entries_negative_control"] += 1
            continue
        if entry["status"] == "ground_truth_unwritten":
            counts["entries_ground_truth_unwritten"] += 1
            continue
        for q in entry["quotes"]:
            counts[f"quotes_{q['status']}"] += 1
    counts["quotes_checked"] = sum(
        v for k, v in counts.items() if k.startswith("quotes_")
    )
    return dict(sorted(counts.items()))


_STATUS_NOTE = {
    "ok": "",
    "dedup_measured": "corpus-wide dedup measurement, not a retrieval target",
    "too_generic": "quote too generic to score cleanly",
    "quote_absent": "quote NOT in the document text -- key quote is paraphrased",
    "split_across_chunks": "quote IS in the document but crosses a chunk boundary",
    "accession_not_indexed": "accession absent from the index",
    "document_not_indexed": "accession indexed, but not this document of it",
    "document_unavailable": "document not found in the fetch ledger",
    "awaiting_quote": "quote not written yet (status NEEDS_QUOTE)",
    "no_quote": "ground truth carries no quote, and no status explains it",
}

#: Statuses that are a declared state of the key, not a finding against it.
#: Flagging these would file planned work as a defect.
_NOT_A_DEFECT = frozenset({"ok", "dedup_measured", "awaiting_quote"})


def _print_report(report: dict[str, Any]) -> None:
    threshold = report["generic_match_threshold"]
    print("=== ANSWER KEY VERIFICATION (report only; the key is not modified) ===")
    print(f"  key   : {report['answer_key']}")
    print(f"  index : {report['index']}")
    print()

    flagged: list[tuple[str, dict[str, Any]]] = []
    for entry in report["results"]:
        if entry["status"] == "negative_control":
            print(f"  {entry['id']:<5} [{entry['type']}] -- no ground truth, "
                  f"as the question type intends")
            continue
        if entry["status"] == "ground_truth_unwritten":
            print(f"  {entry['id']:<5} [{entry['type']}] status="
                  f"{entry['key_status']} -- GROUND TRUTH NOT WRITTEN YET")
            continue
        print(f"  {entry['id']:<5} [{entry['type']}] status={entry['key_status']}")
        for q in entry["quotes"]:
            if q["status"] in ("no_quote", "awaiting_quote"):
                print(f"        (ground truth {q['position']}) "
                      f"{_STATUS_NOTE[q['status']]}")
                if q["status"] not in _NOT_A_DEFECT:
                    flagged.append((entry["id"], q))
                continue
            mark = "  " if q["status"] in _NOT_A_DEFECT else "!!"
            print(
                f"     {mark} {q['ticker'] or '?':<5} {q['accession']}  "
                f"{q['match_count']} match(es)"
                + (f"   <- {_STATUS_NOTE[q['status']]}" if q["status"] != "ok" else "")
            )
            if "distinct_accessions" in q:
                print(f"           distinct accessions: {q['distinct_accessions']}"
                      f"   distinct chunk texts: {q['distinct_texts']}"
                      f"   tickers: {','.join(q['distinct_tickers'])}")
            for m in q["matches"][:10]:
                print(f"           {m['chunk_id']}")
                print(f"              section: {m['section']}   "
                      f"doc: {m['document']}   (key says: {q['key_section']})")
            if len(q["matches"]) > 10:
                print(f"           ... and {len(q['matches']) - 10} more")
            if q["status"] not in _NOT_A_DEFECT:
                flagged.append((entry["id"], q))
        print()

    print(f"=== FLAGGED ({len(flagged)}) ===")
    if not flagged:
        print("  none")
    for entry_id, q in flagged:
        note = _STATUS_NOTE.get(q["status"], q["status"])
        print(f"  {entry_id:<5} {q['status']:<22} {note}")
        if q.get("quote"):
            print(f"        quote ({q['quote_chars']} chars): {q['quote'][:90]!r}")
    print()

    print("=== SUMMARY ===")
    for k, v in report["summary"].items():
        print(f"  {k:<34} {v}")
    print(f"  (a quote matching more than {threshold} chunks is flagged "
          f"'too_generic')")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tools.verify_key",
        description="Check answer-key quotes against the index. Read-only.",
    )
    add_config_args(parser)
    parser.add_argument(
        "--json-only", action="store_true", help="write the JSON report only"
    )
    parser.add_argument(
        "--only", default=None, metavar="IDS",
        help="comma-separated entry ids to check, e.g. q01,q13",
    )
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config, overrides=args.overrides)
        only = (
            {s.strip() for s in args.only.split(",") if s.strip()}
            if args.only
            else None
        )
        out_dir = Path(config.get("evaluate.out_dir"))

        with RunLog(config) as log:
            with log.stage("verify_key") as stage:
                report = verify_key(config, only=only, json_only=args.json_only)
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"key_verification_{log.run_id}.json"
                payload = json.dumps(report, indent=2, default=str)
                out_path.write_text(payload, encoding="utf-8")

                summary = report["summary"]
                stage.count(summary.get("quotes_checked", 0))
                stage.bytes_out(len(payload.encode("utf-8")))
                # No request() calls: local files only.
                stage.note(out_path=str(out_path), **summary)
                for entry in report["results"]:
                    for q in entry.get("quotes", []):
                        if q["status"] not in _NOT_A_DEFECT:
                            stage.error(
                                f"answer key quote: {q['status']}",
                                context={
                                    "id": entry["id"],
                                    "accession": q.get("accession"),
                                    "match_count": q.get("match_count"),
                                },
                            )
        print(f"\nReport written to {out_path}")
        return 0
    except (ConfigError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except ImportError as exc:
        print(f"ERROR: missing dependency ({exc}). Run: uv sync", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
