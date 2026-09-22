"""Evaluation harness: score retrieval against the answer key.

    python -m tools.evaluate
    python -m tools.evaluate --only q13,q19
    python -m tools.evaluate --json-only

Runs every key question through tools.retrieve -- the same code path a user
gets, not a reimplementation -- and scores by question type.

THE HEADLINE IS NOT A RECALL NUMBER
===================================

It is whether the top-1 similarity distribution for answerable questions
separates from the distribution for negative controls. If those two overlap, the
index cannot tell a question it can answer from one it cannot, and every recall
figure above that is describing noise. So the comparison is printed first, both
distributions side by side with every score listed, never collapsed into a
single aggregate.

SCORING IS BY TYPE BECAUSE THE TYPES ASK DIFFERENT QUESTIONS
============================================================

single_fact        did the specific chunk come back
cross_company      did EVERY listed accession come back (a partial set is a miss)
period_over_period did BOTH accessions come back -- coverage, not text
negative_control   no recall at all; the top-1 score IS the measurement
dedup_diagnostic   no recall; how widely is one string spread across filings

The period_over_period rule is the one that matters most. q19's quote is
byte-identical across all eight AGNC 10-Qs, so a text-matching scorer would mark
a retriever correct for returning chunks from six filings nobody asked about.
Only accession coverage separates the finding from a coincidence.

BOUND TO THE INDEX THE KEY WAS VERIFIED AGAINST
===============================================

A key resolves to chunk ids, and those are only valid for one index. Rebuilding
it -- deduplicating, re-chunking, changing the embedder -- changes or removes
them. An evaluator that did not notice would resolve fewer targets, score them
as misses and report a confident number describing nothing. So this refuses to
run unless the live sidecar matches the fingerprint tools.verify_key recorded.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from tools.answerkey import (
    AnswerKeyError,
    MetadataIndex,
    ground_truths,
    index_fingerprint,
    load_key,
    resolve,
)
from tools.config import Config, ConfigError, add_config_args, load_config
from tools.embedders import EmbedderError
from tools.retrieve import Retriever
from tools.runlog import RunLog

__all__ = ["evaluate"]

#: Verification records are per index name. A single fixed filename would make
#: two indexes fight over one record: verifying the key against a second index
#: would silently invalidate the binding for the first, and evaluating it again
#: would refuse. An A/B needs both bindings to coexist.
VERIFICATION_PREFIX = "key_verification_latest"


def verification_record_path(config: Config) -> Path:
    """Where the binding record for THIS index lives."""
    name = str(config.get("index.name"))
    return Path(config.get("evaluate.out_dir")) / f"{VERIFICATION_PREFIX}_{name}.json"


# --------------------------------------------------------------------------
# The binding check
# --------------------------------------------------------------------------


def _check_binding(config: Config, sidecar: dict[str, Any]) -> str:
    """Refuse unless the key was verified against THIS index.

    A warning would not do. This runs in batches, and a warning in a batch is a
    line nobody reads above a number everybody quotes.
    """
    record_path = verification_record_path(config)
    if not record_path.is_file():
        raise ConfigError(
            f"{record_path} not found: the answer key has not been verified "
            f"against index {config.get('index.name')!r}. Run "
            f"`python -m tools.verify_key --set index.name={config.get('index.name')}` "
            f"first -- "
            f"scoring against unverified chunk ids would produce confident "
            f"numbers with nothing behind them."
        )
    recorded = json.loads(record_path.read_text(encoding="utf-8")).get(
        "index_fingerprint"
    )
    live = index_fingerprint(sidecar)
    if recorded != live:
        raise ConfigError(
            f"index fingerprint mismatch.\n"
            f"  key was verified against : {recorded}\n"
            f"  index on disk now is     : {live}\n"
            f"The index has been rebuilt since the key was verified, so the "
            f"chunk ids the key resolves to no longer mean what they meant. "
            f"Re-run `python -m tools.verify_key`, then evaluate again."
        )
    return live


# --------------------------------------------------------------------------
# Per-type scoring
# --------------------------------------------------------------------------


def _first_rank(hits: list[dict[str, Any]], chunk_ids: set[str]) -> int | None:
    for h in hits:
        if h["chunk_id"] in chunk_ids:
            return h["rank"]
    return None


def _accession_rank(hits: list[dict[str, Any]], accession: str) -> int | None:
    for h in hits:
        if h["accession"] == accession:
            return h["rank"]
    return None


def _score_chunk_based(
    hits: list[dict[str, Any]], resolutions: list[Any], recall_at: list[int]
) -> dict[str, Any]:
    """single_fact: did the specific chunk come back?"""
    targets = {cid for r in resolutions for cid in (r.chunk_ids or [])}
    rank = _first_rank(hits, targets)
    top1 = hits[0] if hits else None
    citations = {(r.accession, r.document) for r in resolutions}
    return {
        **{f"recall@{k}": bool(rank is not None and rank <= k) for k in recall_at},
        "rank": rank,
        "top1_correct": bool(top1 and top1["chunk_id"] in targets),
        "citation_correct": bool(
            top1 and (top1["accession"], top1["document"]) in citations
        ),
        "target_chunks": len(targets),
    }


def _score_set_coverage(
    hits: list[dict[str, Any]],
    resolutions: list[Any],
    recall_at: list[int],
    *,
    by_accession: bool,
) -> dict[str, Any]:
    """cross_company and period_over_period: the SET must be covered.

    ``by_accession`` selects what counts as covering a member: any chunk of that
    accession (period_over_period), or a resolved target chunk (cross_company).
    """
    members = []
    for r in resolutions:
        if by_accession:
            rank = _accession_rank(hits, r.accession)
        else:
            rank = _first_rank(hits, set(r.chunk_ids or []))
        members.append(
            {
                "accession": r.accession,
                "ticker": r.ticker,
                "document": r.document,
                "rank": rank,
                # Reported per member and never averaged: q18's two spans are a
                # table row and prose, so a rank gap between them is expected.
                **{f"covered@{k}": bool(rank is not None and rank <= k)
                   for k in recall_at},
            }
        )
    out: dict[str, Any] = {"members": members}
    for k in recall_at:
        out[f"recall@{k}"] = all(m[f"covered@{k}"] for m in members) if members else False
        out[f"covered@{k}"] = sum(1 for m in members if m[f"covered@{k}"])
    out["members_total"] = len(members)
    return out


def _score_target_recall(
    hits: list[dict[str, Any]], resolutions: list[Any], recall_at: list[int]
) -> dict[str, Any]:
    """Secondary metric for period_over_period: did the TARGET chunk come back?

    Coverage alone cannot tell an earned hit from a lucky one where the target
    text is distinctive.

    But where the identifying quote also occurs in OTHER filings, the embedder
    sees near-identical text in several chunks and which of them ranks first
    turns on incidental surrounding words rather than on retrieval quality.
    q19 is the case: its hedging sentence is verbatim in all eight AGNC 10-Qs.
    That is marked, not omitted -- omitting it would read as an absent result
    rather than an inapplicable one.
    """
    spread = max((r.corpus_accessions or 1) for r in resolutions)
    duplicated = max((r.duplicate_text_count or 1) for r in resolutions)
    members = []
    for r in resolutions:
        rank = _first_rank(hits, set(r.chunk_ids or []))
        members.append(
            {
                "accession": r.accession,
                "rank": rank,
                "duplicate_text_count": r.duplicate_text_count,
                "corpus_accessions": r.corpus_accessions,
                **{f"hit@{k}": bool(rank is not None and rank <= k) for k in recall_at},
            }
        )
    out: dict[str, Any] = {
        "meaningful": spread <= 1 and duplicated <= 1,
        "max_corpus_accessions": spread,
        "max_duplicate_text_count": duplicated,
        "members": members,
    }
    if spread > 1:
        out["not_meaningful_because"] = (
            f"the identifying quote occurs in {spread} different accessions; "
            f"which near-identical chunk ranks first is incidental, so chunk "
            f"identity does not measure retrieval quality here"
        )
    elif duplicated > 1:
        out["not_meaningful_because"] = (
            f"a target chunk's text occurs in {duplicated} chunks corpus-wide; "
            f"identical vectors make chunk identity a matter of row order"
        )
    for k in recall_at:
        out[f"recall@{k}"] = (
            all(m[f"hit@{k}"] for m in members) if members else False
        )
    return out


def _score_dedup(hits: list[dict[str, Any]], top_k: int) -> dict[str, Any]:
    window = hits[:top_k]
    texts = {h["text"] for h in window}
    accessions = {h["accession"] for h in window}
    return {
        "window": len(window),
        "distinct_accessions": len(accessions),
        "distinct_texts": len(texts),
        "ratio_accessions_per_text": (
            round(len(accessions) / len(texts), 2) if texts else None
        ),
        "tickers": sorted({h["ticker"] for h in window}),
    }


# --------------------------------------------------------------------------
# Harness
# --------------------------------------------------------------------------


def evaluate(
    config: Config,
    *,
    only: set[str] | None = None,
    json_only: bool = False,
    retriever: Any = None,
) -> dict[str, Any]:
    """Score retrieval against the key.

    ``retriever`` injects an alternative to the dense one -- BM25, or an RRF
    fusion of the two. All three are then scored by THIS code rather than by
    three separate implementations, which is what makes their numbers
    comparable at all.
    """
    key_path = Path(config.get("evaluate.answer_key_path"))
    recall_at = [int(k) for k in config.get("evaluate.recall_at", [5, 20])]
    top_k = int(config.get("retrieve.top_k"))
    section_tickers = set(config.get("evaluate.section_accuracy_tickers", []))
    answerable_types = set(config.get("evaluate.answerable_types", []))
    threshold = int(config.get("evaluate.generic_match_threshold", 3))

    key = load_key(key_path)
    results: list[dict[str, Any]] = []

    with RunLog(config) as log:
        with log.stage("evaluate") as stage:
            if retriever is None:
                retriever = Retriever(config, stage)
            # A retriever may carry a separate binding sidecar: BM25 over the
            # same chunk set is bound by the CHUNK identity, which is what the
            # key's chunk ids depend on, not by an embedder it does not have.
            fingerprint = _check_binding(
                config, getattr(retriever, "binding_sidecar", retriever.sidecar)
            )
            mindex = MetadataIndex(retriever.metadata)

            stage.note(
                answer_key=str(key_path),
                retriever=getattr(retriever, "kind", "dense"),
                index_dir=str(retriever.index_dir),
                index_fingerprint=fingerprint,
                index_size=len(retriever.metadata),
                top_k=top_k,
                recall_at=recall_at,
                section_accuracy_tickers=sorted(section_tickers),
            )

            for entry in key:
                entry_id = str(entry.get("id"))
                if only and entry_id not in only:
                    continue
                qtype = str(entry.get("type"))
                question = entry.get("question")
                if not question:
                    stage.error(
                        "entry has no question; skipped",
                        context={"id": entry_id, "type": qtype},
                    )
                    continue

                hits = retriever.search(question, top_k, stage)
                stage.count()
                resolutions = [
                    resolve(t, mindex, generic_match_threshold=threshold)
                    for t in ground_truths(entry)
                ]
                unresolved = [r for r in resolutions if not r.resolved]
                for r in unresolved:
                    stage.error(
                        f"ground truth did not resolve: {r.status}",
                        context={"id": entry_id, "accession": r.accession},
                    )

                record: dict[str, Any] = {
                    "id": entry_id,
                    "type": qtype,
                    "question": question,
                    "top1_score": hits[0]["score"] if hits else None,
                    "top1_chunk_id": hits[0]["chunk_id"] if hits else None,
                    "top1_ticker": hits[0]["ticker"] if hits else None,
                    "unresolved_ground_truths": len(unresolved),
                    "hits": [
                        {k: h[k] for k in
                         ("rank", "score", "chunk_id", "ticker", "accession",
                          "document", "section")}
                        for h in hits
                    ],
                }

                if qtype == "negative_control":
                    record["scored"] = False
                elif qtype == "dedup_diagnostic":
                    record["scored"] = False
                    record["dedup"] = _score_dedup(hits, top_k)
                elif qtype == "single_fact":
                    record["scored"] = True
                    record["score"] = _score_chunk_based(hits, resolutions, recall_at)
                    record["section"] = _section_result(
                        hits, resolutions, section_tickers
                    )
                elif qtype == "cross_company":
                    record["scored"] = True
                    record["score"] = _score_set_coverage(
                        hits, resolutions, recall_at, by_accession=False
                    )
                    record["section"] = _section_result(
                        hits, resolutions, section_tickers
                    )
                elif qtype == "period_over_period":
                    record["scored"] = True
                    record["score"] = _score_set_coverage(
                        hits, resolutions, recall_at, by_accession=True
                    )
                    record["target_recall"] = _score_target_recall(
                        hits, resolutions, recall_at
                    )
                else:
                    record["scored"] = False
                    stage.error(
                        f"unknown question type {qtype!r}; not scored",
                        context={"id": entry_id},
                    )
                results.append(record)

            report = _aggregate(
                results,
                recall_at=recall_at,
                answerable_types=answerable_types,
                section_tickers=section_tickers,
            )
            report.update(
                {
                    "answer_key": str(key_path),
                    "retriever": getattr(retriever, "kind", "dense"),
                    "index_dir": str(retriever.index_dir),
                    "index_fingerprint": fingerprint,
                    "embedder": (
                        retriever.embedder.describe()
                        if hasattr(retriever, "embedder") else None
                    ),
                    "top_k": top_k,
                    "results": results,
                }
            )

            out_dir = Path(config.get("evaluate.out_dir"))
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{log.run_id}.json"
            payload = json.dumps(report, indent=2, default=str)
            out_path.write_text(payload, encoding="utf-8")
            stage.bytes_out(len(payload.encode("utf-8")))
            report["out_path"] = str(out_path)
            stage.note(
                out_path=str(out_path),
                questions_scored=sum(1 for r in results if r["scored"]),
                **{f"headline_{k}": v
                   for k, v in report["headline"].items()
                   if not isinstance(v, (list, dict))},
            )

    if not json_only:
        _print_report(report)
    return report


def _section_result(
    hits: list[dict[str, Any]], resolutions: list[Any], tickers: set[str]
) -> dict[str, Any]:
    """Did the top hit land in the same section as the target chunk?

    Measured against the INDEX's label for the target, not the key's `section:`
    field -- that field is still TBD for most eligible ground truths, so scoring
    against it would leave the metric unscoreable exactly where it applies.
    """
    eligible = [r for r in resolutions if r.ticker in tickers and r.sections]
    if not eligible or not hits:
        return {"eligible": False}
    expected = {s for r in eligible for s in r.sections}
    return {
        "eligible": True,
        "expected": sorted(expected),
        "observed": str(hits[0]["section"]),
        "correct": str(hits[0]["section"]) in expected,
        "tickers": sorted({r.ticker for r in eligible}),
    }


def _stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "min": None, "median": None, "max": None}
    return {
        "n": len(values),
        "min": round(min(values), 4),
        "median": round(statistics.median(values), 4),
        "max": round(max(values), 4),
        "scores": [round(v, 4) for v in sorted(values, reverse=True)],
    }


def _aggregate(
    results: list[dict[str, Any]],
    *,
    recall_at: list[int],
    answerable_types: set[str],
    section_tickers: set[str],
) -> dict[str, Any]:
    answerable = [
        r["top1_score"] for r in results
        if r["type"] in answerable_types and r["top1_score"] is not None
    ]
    negative = [
        r["top1_score"] for r in results
        if r["type"] == "negative_control" and r["top1_score"] is not None
    ]
    a_stats, n_stats = _stats(answerable), _stats(negative)
    overlap = bool(
        answerable and negative and min(answerable) <= max(negative)
    )
    headline = {
        "answerable": a_stats,
        "negative_control": n_stats,
        "median_separation": (
            round(a_stats["median"] - n_stats["median"], 4)
            if a_stats["median"] is not None and n_stats["median"] is not None
            else None
        ),
        "distributions_overlap": overlap,
        "lowest_answerable": a_stats["min"],
        "highest_negative_control": n_stats["max"],
    }

    by_type: dict[str, Any] = {}
    for qtype in sorted({r["type"] for r in results}):
        rows = [r for r in results if r["type"] == qtype and r["scored"]]
        if not rows:
            by_type[qtype] = {"n": len([r for r in results if r["type"] == qtype]),
                              "scored": False}
            continue
        agg: dict[str, Any] = {"n": len(rows), "scored": True}
        for k in recall_at:
            hits = [r["score"].get(f"recall@{k}") for r in rows]
            agg[f"recall@{k}"] = round(sum(bool(h) for h in hits) / len(hits), 4)
        if qtype in ("single_fact",):
            agg["top1_correct"] = round(
                sum(bool(r["score"]["top1_correct"]) for r in rows) / len(rows), 4)
            agg["citation_correct"] = round(
                sum(bool(r["score"]["citation_correct"]) for r in rows) / len(rows), 4)
        if qtype == "period_over_period":
            meaningful = [r for r in rows if r["target_recall"]["meaningful"]]
            agg["target_recall_meaningful_n"] = len(meaningful)
            for k in recall_at:
                agg[f"target_recall@{k}"] = (
                    round(sum(bool(r["target_recall"][f"recall@{k}"])
                              for r in meaningful) / len(meaningful), 4)
                    if meaningful else None
                )
        by_type[qtype] = agg

    section_rows = [
        r for r in results if r.get("section", {}).get("eligible")
    ]
    section = {
        "tickers_included": sorted(section_tickers),
        "excluded": {
            "AGNC": "chunks are unsectioned (anchor gap)",
            "HON": "Notes text carries part2_item1a_risk_factors (anchor mislabel)",
            "AIG": "Notes text carries part1_item2_mdna (anchor mislabel)",
        },
        "n": len(section_rows),
        "correct": sum(1 for r in section_rows if r["section"]["correct"]),
        "accuracy": (
            round(sum(1 for r in section_rows if r["section"]["correct"])
                  / len(section_rows), 4) if section_rows else None
        ),
    }
    return {"headline": headline, "by_type": by_type, "section_accuracy": section}


def _print_report(report: dict[str, Any]) -> None:
    h = report["headline"]
    print("\n" + "=" * 78)
    print("=== TOP-1 SIMILARITY: ANSWERABLE vs NEGATIVE CONTROL ===")
    print("=" * 78)
    print(f"  {'':<18} {'n':>3} {'min':>8} {'median':>8} {'max':>8}")
    for label in ("answerable", "negative_control"):
        s = h[label]
        print(f"  {label:<18} {s['n']:>3} "
              f"{_f(s['min']):>8} {_f(s['median']):>8} {_f(s['max']):>8}")
    print()
    print(f"  median separation          : {_f(h['median_separation'])}")
    print(f"  lowest answerable          : {_f(h['lowest_answerable'])}")
    print(f"  highest negative control   : {_f(h['highest_negative_control'])}")
    print(f"  distributions overlap      : {h['distributions_overlap']}")
    for label in ("answerable", "negative_control"):
        scores = h[label].get("scores") or []
        print(f"  {label} scores: " + ", ".join(_f(v) for v in scores))

    print("\n=== BY TYPE ===")
    for qtype, agg in report["by_type"].items():
        if not agg.get("scored"):
            print(f"  {qtype:<20} n={agg['n']:<3} not scored on recall")
            continue
        bits = [f"{k}={agg[k]}" for k in agg if k.startswith(("recall@", "top1_", "citation_"))]
        print(f"  {qtype:<20} n={agg['n']:<3} " + "  ".join(bits))
        if qtype == "period_over_period":
            tr = [f"{k}={agg[k]}" for k in agg if k.startswith("target_recall@")]
            print(f"  {'':<20} target-chunk recall (secondary, over "
                  f"{agg['target_recall_meaningful_n']} meaningful): " + "  ".join(tr))

    s = report["section_accuracy"]
    print("\n=== SECTION ACCURACY ===")
    print(f"  included tickers : {', '.join(s['tickers_included'])}")
    print(f"  n={s['n']}  correct={s['correct']}  accuracy={s['accuracy']}")
    print("  EXCLUDED, and why:")
    for ticker, why in s["excluded"].items():
        print(f"    {ticker:<6} {why}")

    print("\n=== PER QUESTION ===")
    for r in report["results"]:
        line = f"  {r['id']:<5} {r['type']:<20} top1={_f(r['top1_score'])}"
        if r["type"] == "negative_control":
            print(line + "   (no recall by construction)")
        elif r["type"] == "dedup_diagnostic":
            d = r["dedup"]
            print(line + f"   accessions={d['distinct_accessions']} "
                         f"texts={d['distinct_texts']} "
                         f"ratio={d['ratio_accessions_per_text']}")
        elif r["type"] == "single_fact":
            sc = r["score"]
            print(line + f"   rank={sc['rank']} "
                         f"r@5={sc['recall@5']} r@20={sc['recall@20']} "
                         f"top1={sc['top1_correct']} cite={sc['citation_correct']}")
        else:
            sc = r["score"]
            cov = f"{sc['covered@20']}/{sc['members_total']}"
            extra = ""
            if r["type"] == "period_over_period":
                tr = r["target_recall"]
                extra = ("   target-recall n/a (duplicated text)"
                         if not tr["meaningful"]
                         else f"   target r@20={tr['recall@20']}")
            print(line + f"   coverage@20={cov} r@5={sc['recall@5']} "
                         f"r@20={sc['recall@20']}{extra}")
            for m in sc["members"]:
                print(f"        {m['accession']}  rank={m['rank']}")

    print(f"\nWrote {report.get('out_path')}")


def _f(v: Any) -> str:
    return "n/a" if v is None else f"{v:.4f}" if isinstance(v, float) else str(v)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tools.evaluate",
        description="Score retrieval against the answer key.",
    )
    add_config_args(parser)
    parser.add_argument("--only", default=None, metavar="IDS",
                        help="comma-separated entry ids, e.g. q13,q19")
    parser.add_argument("--json-only", action="store_true",
                        help="write the JSON report without printing")
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config, overrides=args.overrides)
        only = ({s.strip() for s in args.only.split(",") if s.strip()}
                if args.only else None)
        evaluate(config, only=only, json_only=args.json_only)
        return 0
    except (ConfigError, AnswerKeyError, EmbedderError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except ImportError as exc:
        print(f"ERROR: missing dependency ({exc}). Run: uv sync", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
