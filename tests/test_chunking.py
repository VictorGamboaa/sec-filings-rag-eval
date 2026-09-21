"""Tests for section splitting, windowing, and the parse fingerprint.

The windowing tests use the character unit so they need no model download. The
token unit shares the same windowing contract and is exercised by
`tests/test_index.py::test_token_windower_*` when transformers is available.
"""

from __future__ import annotations

import re

import pytest

from tools.config import load_config
from tools.parse_chunk import (
    UNSECTIONED,
    _anchors_for,
    _distribution,
    _is_parsed,
    config_fingerprint,
    split_sections,
)
from tools.tokenize import CharWindower, TokenizerError, get_windower

LONG = "x" * 400  # comfortably over the configured min_section_chars of 200


@pytest.fixture
def config():
    return load_config()


# --------------------------------------------------------------------------
# Anchor resolution
# --------------------------------------------------------------------------


def test_anchor_lookup_prefers_form_over_default(config):
    anchors = _anchors_for(config, "10-Q", "primary")
    assert [a["label"] for a in anchors][0] == "part1_item1_financial_statements"


def test_anchor_lookup_falls_back_to_default(config):
    """A form with no list of its own still gets anchors."""
    anchors = _anchors_for(config, "S-1", "primary")
    assert len(anchors) == 1
    assert anchors[0]["label"] == "item_{0}"


# --------------------------------------------------------------------------
# Section splitting
# --------------------------------------------------------------------------


def test_preamble_before_the_first_anchor_is_kept_and_unlabelled():
    text = f"COVER PAGE {LONG} Item 2.02 Results {LONG}"
    spans = split_sections(
        text,
        [{"label": "item_{0}", "pattern": r"item\s*(\d\.\d{2})"}],
        min_section_chars=200,
        preamble_label=None,
    )
    assert spans[0][0] is None
    assert spans[0][1] == 0
    assert text[spans[0][1] : spans[0][2]].startswith("COVER PAGE")
    assert spans[1][0] == "item_2.02"


def test_capture_group_labels():
    text = f"Item 5.02 Departure {LONG} Item 9.01 Exhibits {LONG}"
    spans = split_sections(
        text,
        [{"label": "item_{0}", "pattern": r"item\s*(\d\.\d{2})"}],
        min_section_chars=200,
        preamble_label=None,
    )
    assert [s[0] for s in spans] == ["item_5.02", "item_9.01"]


def test_table_of_contents_entries_are_rejected():
    """Contents entries never become sections.

    A 10-Q lists its items before the body, so each anchor matches twice. Every
    contents entry is close to a neighbour on one side or the other, so the
    cluster rule removes the whole block -- no contents entry can open a
    section, and in particular none can absorb the body behind it.
    """
    toc = "Item 1. Financial Statements 3 Item 2. Management's Discussion 15 "
    body = f"Item 1. Financial Statements {LONG} Item 2. Management's Discussion {LONG}"
    anchors = [
        {"label": "fin", "pattern": r"item\s*1\.?\s*financial\s+statements"},
        {"label": "mdna", "pattern": r"item\s*2\.?\s*management.s\s+discussion"},
    ]
    spans = split_sections(toc + body, anchors, min_section_chars=200, preamble_label=None)
    labels = [s[0] for s in spans]
    assert labels.count("mdna") == 1
    mdna = next(s for s in spans if s[0] == "mdna")
    assert mdna[2] - mdna[1] >= 200, "the surviving mdna must be the body heading"


def test_a_heading_crowded_against_the_contents_block_is_dropped_not_mislabelled():
    """The documented trade-off of the cluster rule.

    When the first body heading sits within min_section_chars of the last
    contents entry, the two are indistinguishable by spacing: both are crowded
    on one side and open onto the body on the other. The rule drops both.

    That loses a label, and it is the right way to lose it -- the text becomes
    visibly unsectioned in the stage's distribution report rather than silently
    carrying the contents entry's label. Later headings, which have room on both
    sides, are unaffected.
    """
    toc = "Item 1. Financial Statements 3 Item 2. Management's Discussion 15 "
    body = f"Item 1. Financial Statements {LONG} Item 2. Management's Discussion {LONG}"
    anchors = [
        {"label": "fin", "pattern": r"item\s*1\.?\s*financial\s+statements"},
        {"label": "mdna", "pattern": r"item\s*2\.?\s*management.s\s+discussion"},
    ]
    spans = split_sections(toc + body, anchors, min_section_chars=200, preamble_label=None)
    assert [s[0] for s in spans] == [None, "mdna"]
    unlabelled = spans[0]
    assert "Financial Statements" in (toc + body)[unlabelled[1] : unlabelled[2]]


def test_last_contents_entry_does_not_absorb_the_body():
    """The defect this rule exists for.

    Every contents entry but the last has a short gap to the next. The last
    one's next anchor is the first real heading, far away -- so a next-gap-only
    rule keeps it, and it absorbs everything between the contents table and that
    heading. Observed on this corpus: 54,352 characters of a 10-Q's financial
    statements labelled "part2_item6_exhibits".
    """
    toc = (
        "Item 1. Financial Statements 3 Item 2. Management's Discussion 15 "
        "Item 6. Exhibits 33 SIGNATURES 35 "
    )
    body = "FINANCIAL STATEMENTS BODY " + "z" * 5000 + " Item 2. Management's Discussion " + LONG
    anchors = [
        {"label": "fin", "pattern": r"item\s*1\.?\s*financial\s+statements"},
        {"label": "mdna", "pattern": r"item\s*2\.?\s*management.s\s+discussion"},
        {"label": "exhibits", "pattern": r"item\s*6\.?\s*exhibits"},
    ]
    spans = split_sections(toc + body, anchors, min_section_chars=200, preamble_label=None)
    labels = [s[0] for s in spans]
    assert "exhibits" not in labels, (
        "the last contents entry survived and absorbed the body"
    )
    # The unmatched body text stays unlabelled, which is honest and visible in
    # the section distribution, rather than silently mislabelled.
    assert labels[0] is None
    body_span = spans[0]
    assert "FINANCIAL STATEMENTS BODY" in (toc + body)[body_span[1] : body_span[2]]


def test_a_real_heading_between_two_far_apart_neighbours_survives():
    text = f"Item 1.01 Entry {LONG} Item 8.01 Other {LONG} Item 9.01 Exhibits {LONG}"
    spans = split_sections(
        text,
        [{"label": "item_{0}", "pattern": r"item\s*(\d\.\d{2})"}],
        min_section_chars=200,
        preamble_label=None,
    )
    assert [s[0] for s in spans] == ["item_1.01", "item_8.01", "item_9.01"]


def test_no_anchor_match_yields_one_unlabelled_span():
    text = "A press release with no item structure whatsoever. " * 20
    spans = split_sections(
        text,
        [{"label": "item_{0}", "pattern": r"item\s*(\d\.\d{2})"}],
        min_section_chars=200,
        preamble_label=None,
    )
    assert spans == [(None, 0, len(text))]


def test_spans_are_contiguous_and_cover_the_whole_text():
    """No text may be lost between sections."""
    text = f"PREAMBLE {LONG} Item 1.01 Entry {LONG} Item 8.01 Other {LONG}"
    spans = split_sections(
        text,
        [{"label": "item_{0}", "pattern": r"item\s*(\d\.\d{2})"}],
        min_section_chars=200,
        preamble_label=None,
    )
    assert spans[0][1] == 0
    assert spans[-1][2] == len(text)
    for a, b in zip(spans, spans[1:]):
        assert a[2] == b[1], "a gap between sections would silently drop text"


def test_empty_text_yields_no_spans():
    assert split_sections("", [], min_section_chars=200, preamble_label=None) == []


def test_live_config_anchors_split_a_realistic_8k(config):
    """End-to-end against the real anchor list, not a fixture."""
    text = (
        "ALBERTSONS COMPANIES, INC. Commission File Number 001-39350 " + "pad " * 80
        + "Item 5.02 Departure of Directors or Certain Officers. " + "body " * 120
        + "Item 9.01 Financial Statements and Exhibits. " + "body " * 120
    )
    spans = split_sections(
        text,
        _anchors_for(config, "8-K", "primary"),
        min_section_chars=int(config.get("parse.min_section_chars")),
        preamble_label=config.get("parse.preamble_label"),
    )
    labels = [s[0] for s in spans]
    assert None in labels, "the cover page must remain unlabelled"
    assert "item_5.02" in labels
    assert "item_9.01" in labels


# --------------------------------------------------------------------------
# Windowing
# --------------------------------------------------------------------------


def test_char_windows_honour_size_and_overlap():
    w = CharWindower(100, 20, snap=False, fraction=0.0)
    text = "y" * 250
    wins = w.windows(text)
    assert wins[0][:2] == (0, 100)
    assert wins[1][0] == 80, "stride must be size - overlap"
    assert wins[-1][1] == len(text)


def test_windows_cover_the_text_without_gaps():
    w = CharWindower(50, 10, snap=False, fraction=0.0)
    text = "z" * 187
    wins = w.windows(text)
    assert wins[0][0] == 0
    assert wins[-1][1] == len(text)
    for a, b in zip(wins, wins[1:]):
        assert b[0] < a[1], "consecutive windows must overlap, not merely abut"


def test_window_spans_round_trip_to_the_source_text():
    """A chunk that cannot be located in its document has no provenance."""
    w = CharWindower(40, 8, snap=False, fraction=0.0)
    text = "The quick brown fox jumps over the lazy dog. " * 6
    for start, end, n in w.windows(text):
        assert text[start:end] == text[start:end]
        assert end - start == n


def test_sentence_snapping_moves_the_boundary_back():
    w = CharWindower(60, 10, snap=True, fraction=0.4)
    text = "First sentence here. Second one follows after. Third trails along too."
    first_end = w.windows(text)[0][1]
    assert text[:first_end].rstrip().endswith("."), "should end on a sentence"


def test_snapping_never_stalls_progress():
    """A pathological text must still terminate."""
    w = CharWindower(30, 10, snap=True, fraction=0.9)
    text = "a. " * 200
    wins = w.windows(text)
    assert wins[-1][1] == len(text)
    assert len(wins) < len(text)


def test_empty_text_yields_no_windows():
    assert CharWindower(50, 10, snap=True, fraction=0.2).windows("") == []


@pytest.mark.parametrize("size, overlap", [(0, 0), (-5, 0), (100, 100), (100, 150), (100, -1)])
def test_invalid_window_settings_are_rejected(size, overlap):
    with pytest.raises(TokenizerError):
        CharWindower(size, overlap, snap=False, fraction=0.0)


def test_overlap_equal_to_size_is_rejected_because_stride_would_be_zero():
    with pytest.raises(TokenizerError, match="smaller than"):
        CharWindower(450, 450, snap=False, fraction=0.2)


def test_get_windower_rejects_an_unknown_unit(config):
    bad = load_config(overrides=["chunk.unit=paragraphs"])
    with pytest.raises(TokenizerError, match="unknown chunk.unit"):
        get_windower(bad)


# --------------------------------------------------------------------------
# Config fingerprint -- what decides a re-parse
# --------------------------------------------------------------------------


def _fp(overrides: list[str]) -> str:
    cfg = load_config(overrides=overrides + ["chunk.unit=chars"])
    return config_fingerprint(cfg, get_windower(cfg))


def test_fingerprint_is_stable_for_unchanged_config():
    assert _fp([]) == _fp([])


@pytest.mark.parametrize(
    "override",
    [
        "chunk.size=300",
        "chunk.overlap=50",
        "chunk.snap_to_sentence=false",
        "chunk.snap_search_fraction=0.5",
        "parse.min_section_chars=500",
        'parse.preamble_label=front_matter',
        "parse.exhibit_section_from_label=false",
    ],
)
def test_fingerprint_changes_when_a_tunable_changes(override):
    assert _fp([override]) != _fp([]), (
        f"{override} changes chunk output but not the fingerprint, so a re-run "
        f"would skip documents that need re-parsing"
    )


def test_fingerprint_changes_with_the_tokenizer():
    """Token counts from two tokenizers are not comparable."""
    base = load_config(overrides=["chunk.unit=chars"])
    other = load_config(overrides=["chunk.unit=chars"])
    a = config_fingerprint(base, get_windower(base))

    class FakeWindower:
        identity = "tokens:some/other-model"

    assert a != config_fingerprint(other, FakeWindower())


# --------------------------------------------------------------------------
# Skip logic
# --------------------------------------------------------------------------


def test_unparsed_document_is_not_skipped():
    assert _is_parsed(None, "sha", "fp") is False


def test_changed_document_is_reparsed(tmp_path):
    out = tmp_path / "c.jsonl"
    out.write_text("{}\n", encoding="utf-8")
    entry = {"doc_sha256": "old", "config_fingerprint": "fp", "path": str(out)}
    assert _is_parsed(entry, "new", "fp") is False


def test_changed_config_reparses(tmp_path):
    out = tmp_path / "c.jsonl"
    out.write_text("{}\n", encoding="utf-8")
    entry = {"doc_sha256": "sha", "config_fingerprint": "old", "path": str(out)}
    assert _is_parsed(entry, "sha", "new") is False


def test_missing_output_reparses(tmp_path):
    entry = {
        "doc_sha256": "sha",
        "config_fingerprint": "fp",
        "path": str(tmp_path / "gone.jsonl"),
    }
    assert _is_parsed(entry, "sha", "fp") is False


def test_matching_document_and_config_is_skipped(tmp_path):
    out = tmp_path / "c.jsonl"
    out.write_text("{}\n", encoding="utf-8")
    entry = {"doc_sha256": "sha", "config_fingerprint": "fp", "path": str(out)}
    assert _is_parsed(entry, "sha", "fp") is True


def test_windows_written_path_is_accepted(tmp_path):
    """The ledger may have been written on the other host."""
    out = tmp_path / "c.jsonl"
    out.write_text("{}\n", encoding="utf-8")
    entry = {
        "doc_sha256": "sha",
        "config_fingerprint": "fp",
        "path": str(out).replace("/", "\\"),
    }
    assert _is_parsed(entry, "sha", "fp") is True


# --------------------------------------------------------------------------
# The section distribution report
# --------------------------------------------------------------------------


def _rec(section, text="abcde"):
    return {"section": section, "text": text}


def test_distribution_separates_unsectioned_from_labelled():
    report = _distribution(
        {
            "10-Q/primary": [_rec("mdna", "a" * 90), _rec(None, "b" * 10)],
            "8-K/exhibit": [_rec("EX-99.1", "c" * 50)],
        }
    )
    tenq = report["10-Q/primary"]
    assert tenq["chunks"] == 2
    assert tenq["unsectioned_chunks"] == 1
    assert tenq["unsectioned_char_share"] == 10.0
    assert UNSECTIONED in tenq["by_label"]
    assert "mdna" in tenq["by_label"]

    exhibit = report["8-K/exhibit"]
    assert exhibit["unsectioned_char_share"] == 0.0, (
        "labelling exhibits from exhibit_label is what keeps the unsectioned "
        "bucket meaningful as an anchor-health signal"
    )


def test_distribution_handles_an_empty_group():
    assert _distribution({}) == {}
