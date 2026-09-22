"""Tests for stage 5 retrieval and the evaluation harness.

Uses a stub embedder and a synthetic FAISS index, so no model download. The
scoring rules these pin down are the ones where a plausible-looking alternative
would silently produce wrong numbers:

  * period_over_period scored on ACCESSION COVERAGE, not text -- a text-matching
    scorer marks a retriever correct for returning the right words from the
    wrong filings
  * citation scored on (accession, document) -- accession alone cannot tell an
    8-K shell from its exhibit
  * evaluate refuses to run against an index the key was not verified against
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

import tools.evaluate as ev
import tools.retrieve as rt
from tools.answerkey import MetadataIndex, index_fingerprint, resolve
from tools.config import ConfigError, load_config
from tools.embed_index import INDEX_FILE, METADATA_FILE, SIDECAR_FILE

CONFIG = str(Path(__file__).parent.parent / "inputs" / "config.yaml")
DIM = 8


class StubEmbedder:
    """Deterministic. Query vectors are steered by the fixtures below."""

    name = "stub"
    dim = DIM
    max_batch = 4
    normalized = True
    max_seq_tokens = 512

    def __init__(self) -> None:
        self.queries: list[str] = []
        self.plan: dict[str, np.ndarray] = {}

    @staticmethod
    def basis(i: int) -> np.ndarray:
        v = np.zeros(DIM, dtype="float32")
        v[i % DIM] = 1.0
        return v

    def embed_documents(self, texts, stage=None):
        return np.vstack([self.basis(i) for i in range(len(texts))])

    def embed_query(self, text, stage=None):
        self.queries.append(text)
        return self.plan.get(text, self.basis(0))

    def describe(self):
        return {"provider": "stub", "model": "stub-v1", "dim": DIM,
                "normalized": True, "max_seq_tokens": 512}


def _chunk(i, accession, document, ticker="ACI", section="part1_item1_financial_statements",
           text=None, doc_type="primary"):
    return {
        "chunk_id": f"{accession}:{document}:{i:05d}",
        "cik": "0000000001", "ticker": ticker, "accession": accession,
        "form": "10-Q", "filing_date": "2025-06-14", "period": "2025-06-14",
        "section": section, "chunk_ordinal": i, "doc_type": doc_type,
        "document": document, "exhibit_label": None, "n_tokens": 20,
        "char_span": [0, 10], "text": text or f"chunk {i} of {document}",
    }


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """A synthetic index, sidecar, verification record and answer key."""
    monkeypatch.chdir(tmp_path)
    index_dir = tmp_path / "index" / "main"
    index_dir.mkdir(parents=True)
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    stub = StubEmbedder()
    monkeypatch.setattr(rt, "get_embedder", lambda config, stage=None: stub)

    def build(chunks, key_entries, *, verified=True, sidecar_extra=None):
        import faiss

        (index_dir / METADATA_FILE).write_text(
            "".join(json.dumps(c) + "\n" for c in chunks), encoding="utf-8")
        index = faiss.IndexFlatIP(DIM)
        index.add(np.vstack([StubEmbedder.basis(i) for i in range(len(chunks))]))
        faiss.write_index(index, str(index_dir / INDEX_FILE))
        sidecar = {
            "embedder": stub.describe(), "chunk_fingerprint": "fp-test",
            "dim": DIM, "metric": "ip", "count": len(chunks),
        }
        sidecar.update(sidecar_extra or {})
        (index_dir / SIDECAR_FILE).write_text(json.dumps(sidecar), encoding="utf-8")

        key_path = tmp_path / "key.yaml"
        key_path.write_text(yaml.safe_dump(key_entries, sort_keys=False),
                            encoding="utf-8")
        if verified:
            (eval_dir / f"{ev.VERIFICATION_PREFIX}_main.json").write_text(
                json.dumps({"index_fingerprint": index_fingerprint(sidecar)}),
                encoding="utf-8")

        overrides = [
            f"index.out_dir={tmp_path / 'index'}", "index.name=main",
            f"evaluate.out_dir={eval_dir}", f"evaluate.answer_key_path={key_path}",
            f"retrieve.out_dir={tmp_path / 'ret'}",
            f"run.log_dir={tmp_path / 'runs'}", "retrieve.top_k=5",
        ]
        return load_config(CONFIG, overrides=overrides), stub

    return build, stub


# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------


def test_retriever_uses_embed_query_not_embed_documents(workspace):
    """bge is asymmetric; the query prefix is applied by embed_query alone."""
    build, stub = workspace
    config, _ = build([_chunk(0, "A-1", "a.htm")], [])
    r = rt.Retriever(config)
    r.search("a question", 5)
    assert stub.queries == ["a question"]


def test_retriever_refuses_an_index_built_by_another_model(workspace):
    build, stub = workspace
    config, _ = build(
        [_chunk(0, "A-1", "a.htm")], [],
        sidecar_extra={"embedder": {**stub.describe(), "model": "other-model"}},
    )
    with pytest.raises(Exception, match="meaningless|embedder"):
        rt.Retriever(config)


def test_retriever_refuses_when_metadata_and_index_disagree(workspace, tmp_path):
    build, stub = workspace
    config, _ = build([_chunk(i, "A-1", "a.htm") for i in range(3)], [])
    meta = tmp_path / "index" / "main" / METADATA_FILE
    meta.write_text(meta.read_text(encoding="utf-8").rsplit("\n", 2)[0] + "\n",
                    encoding="utf-8")
    with pytest.raises(ConfigError, match="row-aligned"):
        rt.Retriever(config)


def test_search_returns_rank_score_and_provenance(workspace):
    build, stub = workspace
    config, _ = build([_chunk(i, "A-1", "a.htm") for i in range(3)], [])
    hits = rt.Retriever(config).search("q", 3)
    assert [h["rank"] for h in hits] == [1, 2, 3]
    assert hits[0]["chunk_id"] == "A-1:a.htm:00000"
    for field in ("ticker", "accession", "document", "section", "score", "text"):
        assert field in hits[0]


# --------------------------------------------------------------------------
# The binding check
# --------------------------------------------------------------------------


def test_evaluate_refuses_without_a_verification_record(workspace):
    build, stub = workspace
    config, _ = build([_chunk(0, "A-1", "a.htm")], [], verified=False)
    with pytest.raises(ConfigError, match="not been verified"):
        ev.evaluate(config, json_only=True)


def test_evaluate_refuses_when_the_index_was_rebuilt(workspace, tmp_path):
    """The dedup A/B will rebuild; stale chunk ids must not be scored."""
    build, stub = workspace
    config, _ = build([_chunk(0, "A-1", "a.htm")], [])
    record = tmp_path / "eval" / f"{ev.VERIFICATION_PREFIX}_main.json"
    record.write_text(json.dumps({"index_fingerprint": "stale-fingerprint"}),
                      encoding="utf-8")
    with pytest.raises(ConfigError, match="fingerprint mismatch"):
        ev.evaluate(config, json_only=True)


def test_fingerprint_moves_when_only_the_count_changes():
    """A dedup rebuild leaves the chunking config intact; count betrays it."""
    base = {"embedder": {"model": "m"}, "chunk_fingerprint": "fp",
            "dim": 8, "metric": "ip", "count": 100}
    assert index_fingerprint(base) != index_fingerprint({**base, "count": 90})


# --------------------------------------------------------------------------
# period_over_period: accession coverage, not text
# --------------------------------------------------------------------------


PERIOD_QUOTE = "the primary instruments that we use"


def _period_key(accessions):
    return [{
        "id": "q19", "type": "period_over_period", "status": "verified",
        "question": "did the hedging instruments change?",
        "ground_truth": [
            {"ticker": "AGNC", "accession": a, "document": f"{a}.htm",
             "section": None, "quote": PERIOD_QUOTE}
            for a in accessions
        ],
    }]


def test_period_over_period_is_a_miss_when_only_one_accession_is_covered(workspace):
    build, stub = workspace
    chunks = [
        _chunk(0, "TARGET-1", "TARGET-1.htm", ticker="AGNC", text=PERIOD_QUOTE),
        _chunk(1, "OTHER-1", "OTHER-1.htm", ticker="AGNC", text="filler"),
        _chunk(2, "TARGET-2", "TARGET-2.htm", ticker="AGNC", text=PERIOD_QUOTE),
    ]
    config, stub = build(chunks, _period_key(["TARGET-1", "TARGET-2"]))
    # Steer the query so only TARGET-1 and OTHER-1 are in the top 2.
    stub.plan["did the hedging instruments change?"] = StubEmbedder.basis(0)
    report = ev.evaluate(
        load_config(CONFIG, overrides=[*_ov(config), "retrieve.top_k=2"]),
        json_only=True)
    score = report["results"][0]["score"]
    assert score["recall@5"] is False, "one accession covered is not a hit"
    covered = [m["accession"] for m in score["members"] if m["covered@5"]]
    assert covered == ["TARGET-1"]


def test_period_over_period_is_a_hit_when_both_accessions_are_covered(workspace):
    build, stub = workspace
    chunks = [
        _chunk(0, "TARGET-1", "TARGET-1.htm", ticker="AGNC", text=PERIOD_QUOTE),
        _chunk(1, "TARGET-2", "TARGET-2.htm", ticker="AGNC", text=PERIOD_QUOTE),
    ]
    config, _ = build(chunks, _period_key(["TARGET-1", "TARGET-2"]))
    report = ev.evaluate(config, json_only=True)
    score = report["results"][0]["score"]
    assert score["recall@5"] is True
    assert score["covered@5"] == 2


def test_textually_correct_chunks_from_wrong_accessions_do_not_score(workspace):
    """The q19 shape. This is the whole reason coverage is the metric.

    Six other filings carry byte-identical text. A text-matching scorer would
    call this correct; accession coverage must not.
    """
    build, stub = workspace
    chunks = [_chunk(i, f"DECOY-{i}", f"DECOY-{i}.htm", ticker="AGNC",
                     text=PERIOD_QUOTE) for i in range(4)]
    chunks += [
        _chunk(4, "TARGET-1", "TARGET-1.htm", ticker="AGNC", text=PERIOD_QUOTE),
        _chunk(5, "TARGET-2", "TARGET-2.htm", ticker="AGNC", text=PERIOD_QUOTE),
    ]
    config, _ = build(chunks, _period_key(["TARGET-1", "TARGET-2"]))
    report = ev.evaluate(
        load_config(CONFIG, overrides=[*_ov(config), "retrieve.top_k=4"]),
        json_only=True)
    result = report["results"][0]
    assert result["score"]["recall@5"] is False, (
        "decoy chunks with identical text must not satisfy accession coverage"
    )
    # ...and the secondary metric must declare itself inapplicable, not absent.
    # The condition is the quote's SPREAD ACROSS ACCESSIONS, not identical chunk
    # texts: q19's real shape is one sentence recurring in eight filings whose
    # surrounding text differs, so a same-chunk-text rule would never trip.
    tr = result["target_recall"]
    assert tr["meaningful"] is False
    assert tr["max_corpus_accessions"] == 6
    assert "different accessions" in tr["not_meaningful_because"]


def test_target_recall_is_meaningful_when_the_text_is_distinctive(workspace):
    build, stub = workspace
    chunks = [
        _chunk(0, "T-1", "T-1.htm", ticker="ACI", text="a distinctive early span"),
        _chunk(1, "T-2", "T-2.htm", ticker="ACI", text="a distinctive later span"),
    ]
    key = [{
        "id": "q17", "type": "period_over_period", "status": "verified",
        "question": "how did it change?",
        "ground_truth": [
            {"ticker": "ACI", "accession": "T-1", "document": "T-1.htm",
             "section": "TBD", "quote": "distinctive early span"},
            {"ticker": "ACI", "accession": "T-2", "document": "T-2.htm",
             "section": "TBD", "quote": "distinctive later span"},
        ],
    }]
    config, _ = build(chunks, key)
    tr = ev.evaluate(config, json_only=True)["results"][0]["target_recall"]
    assert tr["meaningful"] is True
    assert tr["max_corpus_accessions"] == 1
    assert tr["recall@5"] is True


def test_per_accession_ranks_are_reported_separately(workspace):
    """q18's two spans are tabular and prose; a rank gap is expected."""
    build, stub = workspace
    chunks = [
        _chunk(0, "T-1", "T-1.htm", ticker="AIG", text="tabular 48.4 %"),
        _chunk(1, "PAD", "PAD.htm", ticker="AIG", text="pad"),
        _chunk(2, "T-2", "T-2.htm", ticker="AIG", text="prose 5.6 percent"),
    ]
    key = [{
        "id": "q18", "type": "period_over_period", "status": "verified",
        "question": "ownership change?",
        "ground_truth": [
            {"ticker": "AIG", "accession": "T-1", "document": "T-1.htm",
             "section": "x", "quote": "tabular 48.4 %"},
            {"ticker": "AIG", "accession": "T-2", "document": "T-2.htm",
             "section": "x", "quote": "prose 5.6 percent"},
        ],
    }]
    config, _ = build(chunks, key)
    members = ev.evaluate(config, json_only=True)["results"][0]["score"]["members"]
    assert [m["accession"] for m in members] == ["T-1", "T-2"]
    assert members[0]["rank"] != members[1]["rank"]
    assert all("rank" in m for m in members)


# --------------------------------------------------------------------------
# cross_company and citation
# --------------------------------------------------------------------------


def test_cross_company_partial_coverage_is_a_miss(workspace):
    build, stub = workspace
    chunks = [
        _chunk(0, "ACI-1", "aci.htm", ticker="ACI", text="the statute"),
        _chunk(1, "BCPC-1", "bcpc.htm", ticker="BCPC", text="the statute"),
    ]
    key = [{
        "id": "q13", "type": "cross_company", "status": "verified",
        "question": "who disclosed the statute?",
        "ground_truth": [
            {"ticker": "ACI", "accession": "ACI-1", "document": "aci.htm",
             "section": "TBD", "quote": "the statute"},
            {"ticker": "BCPC", "accession": "BCPC-1", "document": "bcpc.htm",
             "section": "TBD", "quote": "the statute"},
        ],
    }]
    config, _ = build(chunks, key)
    full = ev.evaluate(config, json_only=True)["results"][0]["score"]
    assert full["recall@5"] is True and full["covered@5"] == 2

    partial = ev.evaluate(
        load_config(CONFIG, overrides=[*_ov(config), "retrieve.top_k=1"]),
        json_only=True)["results"][0]["score"]
    assert partial["recall@5"] is False, "a partial set is a miss"


def test_citation_distinguishes_a_primary_from_its_exhibit(workspace):
    """Same accession, different document. Accession alone cannot tell them apart."""
    build, stub = workspace
    chunks = [
        _chunk(0, "A-1", "shell.htm", ticker="HON", text="see exhibit 99.1"),
        _chunk(1, "A-1", "ex99.htm", ticker="HON", text="outstanding results",
               doc_type="exhibit"),
    ]
    key = [{
        "id": "q27", "type": "single_fact", "status": "verified",
        "question": "what did the CEO say?",
        "ground_truth": {"ticker": "HON", "accession": "A-1",
                         "document": "ex99.htm", "section": "EX-99",
                         "quote": "outstanding results"},
    }]
    config, _ = build(chunks, key)
    # Top hit is the SHELL (basis 0); the ground truth lives in the exhibit.
    score = ev.evaluate(config, json_only=True)["results"][0]["score"]
    assert score["citation_correct"] is False
    assert score["top1_correct"] is False
    assert score["rank"] == 2, "the exhibit chunk is still recalled, at rank 2"


# --------------------------------------------------------------------------
# negative controls, dedup, section accuracy, headline
# --------------------------------------------------------------------------


def test_negative_control_records_a_score_and_no_recall(workspace):
    build, stub = workspace
    key = [{"id": "q21", "type": "negative_control", "status": "verified",
            "question": "what did Costco report?", "ground_truth": None}]
    config, _ = build([_chunk(0, "A-1", "a.htm")], key)
    r = ev.evaluate(config, json_only=True)["results"][0]
    assert r["scored"] is False
    assert r["top1_score"] is not None
    assert "score" not in r


def test_dedup_diagnostic_counts_accessions_and_texts(workspace):
    build, stub = workspace
    chunks = [_chunk(i, f"A-{i}", f"a{i}.htm", ticker="HON", text="boilerplate")
              for i in range(4)]
    key = [{"id": "q26", "type": "dedup_diagnostic", "status": "verified",
            "question": "which securities are registered?",
            "ground_truth": {"ticker": "HON", "accession": "ANY",
                             "document": "ANY", "section": None,
                             "quote": "boilerplate"}}]
    config, _ = build(chunks, key)
    d = ev.evaluate(config, json_only=True)["results"][0]["dedup"]
    assert d["distinct_accessions"] == 4
    assert d["distinct_texts"] == 1
    assert d["ratio_accessions_per_text"] == 4.0


def test_section_accuracy_excludes_untrusted_tickers(workspace):
    """HON/AIG labels are confidently wrong; AGNC has none."""
    build, stub = workspace
    chunks = [
        _chunk(0, "H-1", "h.htm", ticker="HON", section="part2_item1a_risk_factors",
               text="honeywell note text"),
    ]
    key = [{"id": "q02", "type": "single_fact", "status": "verified",
            "question": "hon?",
            "ground_truth": {"ticker": "HON", "accession": "H-1",
                             "document": "h.htm",
                             "section": "part2_item1a_risk_factors",
                             "quote": "honeywell note text"}}]
    config, _ = build(chunks, key)
    report = ev.evaluate(config, json_only=True)
    assert report["results"][0]["section"]["eligible"] is False
    assert report["section_accuracy"]["n"] == 0
    assert "AGNC" in report["section_accuracy"]["excluded"]


def test_section_accuracy_works_despite_a_TBD_key_field(workspace):
    """Measured against the index's label, not the key's section: field."""
    build, stub = workspace
    chunks = [_chunk(0, "A-1", "a.htm", ticker="ACI", text="albertsons text")]
    key = [{"id": "q05", "type": "single_fact", "status": "verified",
            "question": "aci?",
            "ground_truth": {"ticker": "ACI", "accession": "A-1",
                             "document": "a.htm", "section": "TBD",
                             "quote": "albertsons text"}}]
    config, _ = build(chunks, key)
    report = ev.evaluate(config, json_only=True)
    sec = report["results"][0]["section"]
    assert sec["eligible"] is True
    assert sec["expected"] == ["part1_item1_financial_statements"]
    assert sec["correct"] is True
    assert report["section_accuracy"]["accuracy"] == 1.0


def test_headline_compares_answerable_against_negative_controls(workspace):
    build, stub = workspace
    chunks = [_chunk(0, "A-1", "a.htm", ticker="ACI", text="answerable text"),
              _chunk(1, "A-2", "b.htm", ticker="ACI", text="other")]
    key = [
        {"id": "q05", "type": "single_fact", "status": "verified",
         "question": "answerable?",
         "ground_truth": {"ticker": "ACI", "accession": "A-1",
                          "document": "a.htm", "section": "TBD",
                          "quote": "answerable text"}},
        {"id": "q21", "type": "negative_control", "status": "verified",
         "question": "unanswerable?", "ground_truth": None},
    ]
    config, stub = build(chunks, key)
    stub.plan["answerable?"] = StubEmbedder.basis(0)
    stub.plan["unanswerable?"] = np.array(
        [0.6, 0.8, 0, 0, 0, 0, 0, 0], dtype="float32")
    h = ev.evaluate(config, json_only=True)["headline"]
    assert h["answerable"]["n"] == 1 and h["negative_control"]["n"] == 1
    assert h["median_separation"] is not None
    assert "distributions_overlap" in h
    assert h["answerable"]["scores"] and h["negative_control"]["scores"]


def test_dedup_is_excluded_from_both_headline_distributions(workspace):
    build, stub = workspace
    key = [{"id": "q26", "type": "dedup_diagnostic", "status": "verified",
            "question": "boilerplate?",
            "ground_truth": {"ticker": "HON", "accession": "ANY",
                             "document": "ANY", "section": None,
                             "quote": "x"}}]
    config, _ = build([_chunk(0, "A-1", "a.htm", text="x")], key)
    h = ev.evaluate(config, json_only=True)["headline"]
    assert h["answerable"]["n"] == 0
    assert h["negative_control"]["n"] == 0


def _ov(config):
    """Rebuild the override list from a config already constructed."""
    return [
        f"index.out_dir={Path(config.get('index.out_dir'))}",
        f"index.name={config.get('index.name')}",
        f"evaluate.out_dir={Path(config.get('evaluate.out_dir'))}",
        f"evaluate.answer_key_path={Path(config.get('evaluate.answer_key_path'))}",
        f"retrieve.out_dir={Path(config.get('retrieve.out_dir'))}",
        f"run.log_dir={Path(config.get('run.log_dir'))}",
    ]


def test_verification_records_are_per_index(workspace, tmp_path):
    """An A/B needs both bindings to coexist.

    One fixed filename would mean verifying the key against a second index
    silently invalidated the binding for the first, so evaluating the first
    again would refuse -- in the middle of the comparison it was built for.
    """
    build, stub = workspace
    config, _ = build([_chunk(0, "A-1", "a.htm")], [])
    assert ev.verification_record_path(config).name.endswith("_main.json")

    other = load_config(CONFIG, overrides=[*_ov(config), "index.name=main_200"])
    assert ev.verification_record_path(other).name.endswith("_main_200.json")
    assert ev.verification_record_path(config) != ev.verification_record_path(other)

    # The record written for `main` satisfies `main`...
    ev.evaluate(config, json_only=True)
    # ...and is not consulted for any other index: removing it refuses, proving
    # the check reads the per-index path rather than any record that exists.
    ev.verification_record_path(config).unlink()
    with pytest.raises(ConfigError, match="not been verified"):
        ev.evaluate(config, json_only=True)
