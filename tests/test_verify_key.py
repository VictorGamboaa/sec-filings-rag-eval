"""Tests for answer-key verification.

The classification of a zero-match result is what these mostly cover: "the key's
quote is wrong" and "the key is right and the chunker split it" are different
findings with different fixes, and a tool that reported both as 0 would leave
that distinction to guesswork.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from tools.config import load_config
from tools.verify_key import _ground_truths, normalize, verify_key

#: The fixtures chdir into tmp_path, so the config must be addressed absolutely.
CONFIG = str(Path(__file__).parent.parent / "inputs" / "config.yaml")


# --------------------------------------------------------------------------
# Normalization
# --------------------------------------------------------------------------


def test_normalize_collapses_whitespace_only():
    assert normalize("the  Company\n  acquired   100%") == "the Company acquired 100%"
    # Nothing else is touched: case and punctuation must survive, or a
    # near-match would be reported as a match.
    assert normalize("The Company's") == "The Company's"
    assert normalize("  leading and trailing  ") == "leading and trailing"


def test_normalization_is_not_fuzzy_matching():
    """A quote differing by more than whitespace must not match."""
    assert normalize("colour") != normalize("color")
    assert normalize("The Company") != normalize("the company")


# --------------------------------------------------------------------------
# ground_truth shapes
# --------------------------------------------------------------------------


def test_ground_truth_accepts_a_mapping():
    assert _ground_truths({"ground_truth": {"quote": "x"}}) == [{"quote": "x"}]


def test_ground_truth_accepts_a_list():
    gts = _ground_truths({"ground_truth": [{"quote": "a"}, {"quote": "b"}]})
    assert len(gts) == 2


def test_negative_control_has_no_ground_truth_and_that_is_not_a_defect():
    assert _ground_truths({"ground_truth": None}) == []
    assert _ground_truths({}) == []


def test_empty_list_ground_truth():
    assert _ground_truths({"ground_truth": []}) == []


# --------------------------------------------------------------------------
# End-to-end classification
# --------------------------------------------------------------------------

ACC = "0000000001-26-000001"
DOC = "doc.htm"


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """A fake index, fetch ledger and source document under tmp_path."""
    monkeypatch.chdir(tmp_path)
    index_dir = tmp_path / "index" / "main"
    index_dir.mkdir(parents=True)
    filings = tmp_path / "filings"
    filings.mkdir()

    # Source document: two sentences that the chunker splits between.
    doc_path = filings / "doc.html"
    doc_path.write_text(
        "<p>The Company acquired 100% of Sundyne, a maker of pumps. "
        "Consideration was 2,160 million dollars in cash.</p>",
        encoding="utf-8",
    )
    ledger = tmp_path / "fetched.csv"
    with ledger.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=[
                "accession", "document", "doc_type", "path", "status",
                "bytes", "content_type", "sha256", "fetched_at",
            ],
            lineterminator="\n",
        )
        w.writeheader()
        w.writerow({
            "accession": ACC, "document": DOC, "doc_type": "primary",
            "path": str(doc_path), "status": "ok", "bytes": doc_path.stat().st_size,
            "content_type": "text/html", "sha256": "x", "fetched_at": "now",
        })

    def write_chunks(chunks):
        (index_dir / "metadata.jsonl").write_text(
            "".join(json.dumps(c) + "\n" for c in chunks), encoding="utf-8"
        )

    def write_key(entries):
        import yaml

        path = tmp_path / "key.yaml"
        path.write_text(yaml.safe_dump(entries, sort_keys=False), encoding="utf-8")
        return path

    overrides = [
        f"index.out_dir={tmp_path / 'index'}",
        "index.name=main",
        f"fetch.ledger_path={ledger}",
        f"run.log_dir={tmp_path / 'runs'}",
        f"evaluate.out_dir={tmp_path / 'eval'}",
    ]
    return write_chunks, write_key, overrides


def _chunk(cid, text, section="part1_item1_financial_statements", accession=ACC):
    return {
        "chunk_id": cid, "cik": "0000000001", "ticker": "TEST",
        "accession": accession, "form": "10-Q", "filing_date": "2026-01-02",
        "period": "2026-01-01", "section": section, "chunk_ordinal": 0,
        "doc_type": "primary", "document": DOC, "exhibit_label": None,
        "n_tokens": 10, "char_span": [0, len(text)], "text": text,
    }


def _entry(quote, accession=ACC, qid="q01"):
    return {
        "id": qid, "type": "single_fact", "author": "claude",
        "question": "?",
        "ground_truth": {
            "ticker": "TEST", "accession": accession, "document": DOC,
            "section": "VERIFY", "quote": quote,
        },
    }


def _run(workspace, chunks, entries, extra=()):
    write_chunks, write_key, overrides = workspace
    write_chunks(chunks)
    key_path = write_key(entries)
    config = load_config(
        CONFIG,
        overrides=list(overrides) + [f"evaluate.answer_key_path={key_path}"] + list(extra),
    )
    return verify_key(config, json_only=True)


def test_quote_found_in_one_chunk_reports_chunk_id_and_section(workspace):
    report = _run(
        workspace,
        [_chunk("c:doc.htm:00000", "The Company acquired 100% of Sundyne, a maker of pumps.")],
        [_entry("acquired 100% of Sundyne")],
    )
    q = report["results"][0]["quotes"][0]
    assert q["status"] == "ok"
    assert q["match_count"] == 1
    assert q["matches"][0]["chunk_id"] == "c:doc.htm:00000"
    assert q["matches"][0]["section"] == "part1_item1_financial_statements"


def test_accession_must_match_not_just_the_text(workspace):
    """A quote found in another filing is not a match."""
    report = _run(
        workspace,
        [_chunk("other:doc.htm:00000", "acquired 100% of Sundyne", accession="9999999999-99-999999")],
        [_entry("acquired 100% of Sundyne")],
    )
    assert report["results"][0]["quotes"][0]["status"] == "accession_not_indexed"
    assert report["results"][0]["quotes"][0]["match_count"] == 0


def test_more_than_threshold_matches_is_flagged_too_generic(workspace):
    chunks = [_chunk(f"c:doc.htm:{i:05d}", "the Company") for i in range(4)]
    report = _run(workspace, chunks, [_entry("the Company")])
    q = report["results"][0]["quotes"][0]
    assert q["match_count"] == 4
    assert q["status"] == "too_generic"


def test_threshold_is_configurable(workspace):
    chunks = [_chunk(f"c:doc.htm:{i:05d}", "the Company") for i in range(4)]
    report = _run(
        workspace, chunks, [_entry("the Company")],
        extra=["evaluate.generic_match_threshold=5"],
    )
    assert report["results"][0]["quotes"][0]["status"] == "ok"


def test_quote_absent_from_the_document_is_distinguished(workspace):
    """The key's quote was paraphrased -- the key is wrong."""
    report = _run(
        workspace,
        [_chunk("c:doc.htm:00000", "The Company acquired 100% of Sundyne, a maker of pumps.")],
        [_entry("the Company purchased all of Sundyne")],
    )
    q = report["results"][0]["quotes"][0]
    assert q["match_count"] == 0
    assert q["status"] == "quote_absent"


def test_quote_split_across_chunks_is_distinguished(workspace):
    """The key is right; the chunker split it. A different finding entirely."""
    report = _run(
        workspace,
        [
            _chunk("c:doc.htm:00000", "The Company acquired 100% of Sundyne, a maker of"),
            _chunk("c:doc.htm:00001", "pumps. Consideration was 2,160 million dollars in cash."),
        ],
        [_entry("a maker of pumps")],
    )
    q = report["results"][0]["quotes"][0]
    assert q["match_count"] == 0
    assert q["status"] == "split_across_chunks", (
        "a quote present in the document but crossing a chunk boundary must not "
        "be reported the same way as a quote the document does not contain"
    )


def test_whitespace_differences_do_not_cause_a_false_negative(workspace):
    report = _run(
        workspace,
        [_chunk("c:doc.htm:00000", "The Company   acquired\n100% of Sundyne")],
        [_entry("The Company acquired 100% of Sundyne")],
    )
    assert report["results"][0]["quotes"][0]["status"] == "ok"


def test_list_ground_truth_checks_every_member(workspace):
    entry = {
        "id": "q13", "type": "cross_company", "author": "claude", "question": "?",
        "ground_truth": [
            {"ticker": "A", "accession": ACC, "document": DOC, "section": "VERIFY",
             "quote": "acquired 100% of Sundyne"},
            {"ticker": "B", "accession": ACC, "document": DOC, "section": "VERIFY",
             "quote": "text that is definitely not present anywhere"},
        ],
    }
    report = _run(
        workspace,
        [_chunk("c:doc.htm:00000", "The Company acquired 100% of Sundyne, a maker of pumps.")],
        [entry],
    )
    statuses = [q["status"] for q in report["results"][0]["quotes"]]
    assert statuses == ["ok", "quote_absent"]


def test_negative_control_is_not_reported_as_missing_work(workspace):
    """A negative control is SUPPOSED to have no ground truth."""
    report = _run(
        workspace,
        [_chunk("c:doc.htm:00000", "anything")],
        [{"id": "q21", "type": "negative_control", "author": "claude",
          "question": "?", "ground_truth": None}],
    )
    assert report["results"][0]["status"] == "negative_control"
    assert report["summary"]["entries_negative_control"] == 1


def test_unwritten_entry_is_not_reported_as_a_negative_control(workspace):
    """An empty ground truth on any other type is work outstanding.

    Merging the two would say an unwritten question is fine by design.
    """
    report = _run(
        workspace,
        [_chunk("c:doc.htm:00000", "anything")],
        [{"id": "q14", "type": "cross_company", "author": "victor",
          "question": "TBD", "ground_truth": []}],
    )
    assert report["results"][0]["status"] == "ground_truth_unwritten"
    assert report["summary"]["entries_ground_truth_unwritten"] == 1


def test_ground_truth_without_a_quote_is_reported(workspace):
    entry = {
        "id": "q14", "type": "cross_company", "author": "victor", "question": "TBD",
        "ground_truth": [{"ticker": "A", "accession": ACC, "section": "VERIFY"}],
    }
    report = _run(workspace, [_chunk("c:doc.htm:00000", "x")], [entry])
    assert report["results"][0]["quotes"][0]["status"] == "no_quote"


def test_only_filter_restricts_which_entries_are_checked(workspace):
    write_chunks, write_key, overrides = workspace
    write_chunks([_chunk("c:doc.htm:00000", "acquired 100% of Sundyne")])
    key_path = write_key([_entry("acquired 100% of Sundyne", qid="q01"),
                          _entry("acquired 100% of Sundyne", qid="q02")])
    config = load_config(
        CONFIG, overrides=list(overrides) + [f"evaluate.answer_key_path={key_path}"]
    )
    report = verify_key(config, only={"q02"}, json_only=True)
    assert [r["id"] for r in report["results"]] == ["q02"]


def test_the_key_file_is_never_modified(workspace):
    write_chunks, write_key, overrides = workspace
    write_chunks([_chunk("c:doc.htm:00000", "acquired 100% of Sundyne")])
    key_path = write_key([_entry("a quote that is absent")])
    before = key_path.read_bytes()
    config = load_config(
        CONFIG, overrides=list(overrides) + [f"evaluate.answer_key_path={key_path}"]
    )
    verify_key(config, json_only=True)
    assert key_path.read_bytes() == before, "the key must never be rewritten"


# --------------------------------------------------------------------------
# v2: document scoping, ANY, and declared status
# --------------------------------------------------------------------------


def _exhibit_chunk(cid, text, document="exhibit99.htm"):
    c = _chunk(cid, text, section="EX-99.1")
    c["doc_type"] = "exhibit"
    c["document"] = document
    c["exhibit_label"] = "EX-99.1"
    return c


def test_ground_truth_in_an_exhibit_does_not_match_the_primary(workspace):
    """The distinction the exhibit expansion exists to make measurable.

    A filing's accession covers its primary AND its exhibits. Scoping on
    accession alone would let a match in the 8-K shell satisfy a ground truth
    that lives in the press release, which is exactly the confusion this key
    entry is designed to detect.
    """
    entry = {
        "id": "q27", "type": "single_fact", "status": "UNVERIFIED", "question": "?",
        "ground_truth": {
            "ticker": "TEST", "accession": ACC, "document": "exhibit99.htm",
            "section": "TBD", "quote": "delivered outstanding results",
        },
    }
    report = _run(
        workspace,
        [
            # Same accession, same text, but the PRIMARY document.
            _chunk("p:doc.htm:00000", "The company delivered outstanding results"),
            _exhibit_chunk("e:exhibit99.htm:00000", "We delivered outstanding results in Q2"),
        ],
        [entry],
    )
    q = report["results"][0]["quotes"][0]
    assert q["status"] == "ok"
    assert q["match_count"] == 1, "the primary-document chunk must not count"
    assert q["matches"][0]["document"] == "exhibit99.htm"
    assert q["matches"][0]["doc_type"] == "exhibit"


def test_document_indexed_but_quote_only_in_the_other_document(workspace):
    entry = {
        "id": "q27", "type": "single_fact", "status": "UNVERIFIED", "question": "?",
        "ground_truth": {
            "ticker": "TEST", "accession": ACC, "document": "exhibit99.htm",
            "section": "TBD", "quote": "only in the primary",
        },
    }
    report = _run(
        workspace,
        [
            _chunk("p:doc.htm:00000", "this text is only in the primary document"),
            _exhibit_chunk("e:exhibit99.htm:00000", "unrelated exhibit text"),
        ],
        [entry],
    )
    assert report["results"][0]["quotes"][0]["status"] == "quote_absent"


def test_named_document_absent_from_the_index_is_its_own_status(workspace):
    entry = {
        "id": "qx", "type": "single_fact", "status": "UNVERIFIED", "question": "?",
        "ground_truth": {
            "ticker": "TEST", "accession": ACC, "document": "never-fetched.htm",
            "section": "TBD", "quote": "anything",
        },
    }
    report = _run(workspace, [_chunk("p:doc.htm:00000", "anything at all")], [entry])
    assert report["results"][0]["quotes"][0]["status"] == "document_not_indexed"


def test_any_accession_searches_the_whole_corpus_and_counts_accessions(workspace):
    """The dedup diagnostic: breadth is the measurement."""
    entry = {
        "id": "q26", "type": "dedup_diagnostic", "status": "UNVERIFIED",
        "question": "?",
        "ground_truth": {
            "ticker": "TEST", "accession": "ANY", "document": "ANY",
            "section": "TBD", "quote": "855 S. Mint Street",
        },
    }
    chunks = [
        _chunk("a:doc.htm:00000", "855 S. Mint Street Charlotte", accession="0000000001-26-000001"),
        _chunk("b:doc.htm:00000", "855 S. Mint Street Charlotte", accession="0000000002-26-000002"),
        _chunk("c:doc.htm:00000", "855 S. Mint Street Charlotte", accession="0000000003-26-000003"),
        _chunk("d:doc.htm:00000", "855 S. Mint Street, suite 100", accession="0000000004-26-000004"),
        _chunk("e:doc.htm:00000", "something else entirely", accession="0000000005-26-000005"),
    ]
    report = _run(workspace, chunks, [entry])
    q = report["results"][0]["quotes"][0]
    assert q["match_count"] == 4
    assert q["distinct_accessions"] == 4
    # Three chunks share identical text; the fourth differs.
    assert q["distinct_texts"] == 2


def test_any_accession_is_never_flagged_too_generic(workspace):
    """Matching many chunks is the point of a dedup diagnostic, not a defect."""
    entry = {
        "id": "q26", "type": "dedup_diagnostic", "status": "UNVERIFIED",
        "question": "?",
        "ground_truth": {
            "ticker": "TEST", "accession": "ANY", "document": "ANY",
            "section": "TBD", "quote": "boilerplate",
        },
    }
    chunks = [
        _chunk(f"c{i}:doc.htm:00000", "boilerplate", accession=f"000000000{i}-26-00000{i}")
        for i in range(8)
    ]
    report = _run(workspace, chunks, [entry])
    q = report["results"][0]["quotes"][0]
    assert q["match_count"] == 8
    assert q["status"] == "dedup_measured", "breadth must not read as 'too generic'"
    assert "q26" not in [
        e["id"] for e in report["results"] if e["status"] != "checked"
    ]


def test_needs_quote_is_awaiting_not_failing(workspace):
    """A declared NEEDS_QUOTE is planned work, not a defect."""
    entry = {
        "id": "q01", "type": "single_fact", "status": "NEEDS_QUOTE", "question": "?",
        "ground_truth": {
            "ticker": "TEST", "accession": ACC, "document": DOC,
            "section": "TBD", "quote": None,
        },
    }
    report = _run(workspace, [_chunk("p:doc.htm:00000", "x")], [entry])
    q = report["results"][0]["quotes"][0]
    assert q["status"] == "awaiting_quote"
    assert report["summary"]["quotes_awaiting_quote"] == 1


def test_missing_quote_without_a_declared_status_is_still_flagged(workspace):
    """Only NEEDS_QUOTE excuses a null quote."""
    entry = {
        "id": "qz", "type": "single_fact", "status": "verified", "question": "?",
        "ground_truth": {
            "ticker": "TEST", "accession": ACC, "document": DOC,
            "section": "TBD", "quote": None,
        },
    }
    report = _run(workspace, [_chunk("p:doc.htm:00000", "x")], [entry])
    assert report["results"][0]["quotes"][0]["status"] == "no_quote"


def test_summary_counts_the_keys_declared_statuses(workspace):
    entries = [
        {"id": "a", "type": "single_fact", "status": "verified", "question": "?",
         "ground_truth": {"accession": ACC, "document": DOC, "quote": "hello",
                          "ticker": "TEST", "section": "x"}},
        {"id": "b", "type": "negative_control", "status": "verified",
         "question": "?", "ground_truth": None},
        {"id": "c", "type": "period_over_period", "status": "TO_WRITE",
         "question": "?", "ground_truth": []},
    ]
    report = _run(workspace, [_chunk("p:doc.htm:00000", "hello there")], entries)
    s = report["summary"]
    assert s["key_status_verified"] == 2
    assert s["key_status_TO_WRITE"] == 1
