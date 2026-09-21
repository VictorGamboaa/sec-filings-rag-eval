"""Tests for the corpus audit's matching rules.

Both rules here were found wrong by a real run and are pinned so they cannot
regress: each one silently overstated the size of the corpus gap, which for an
audit is the failure that matters.
"""

from __future__ import annotations

import re

import pytest

from tools.audit_corpus import _is_resolved, _split_item_narrative
from tools.config import load_config


@pytest.fixture
def exhibit_re():
    return re.compile(load_config().get("audit.exhibit_pattern"), re.I)


# --------------------------------------------------------------------------
# The exhibit reference pattern
# --------------------------------------------------------------------------


def test_pattern_does_not_match_inside_a_filename(exhibit_re):
    """The bug this guards against.

    Without a trailing word boundary, "ex991-pressrelease.htm" matches as
    "ex" + "99" -- the trailing "1" unclaimed because no dot follows -- yielding
    a bare "99" reference that can never resolve against a label of "99.1".
    On the real corpus that reported 73 phantom unresolved references.
    """
    assert exhibit_re.findall("see ex991-pressreleasebod.htm attached") == []
    assert exhibit_re.findall("ex992-opioid.htm") == []


@pytest.mark.parametrize(
    "text, expected",
    [
        ("attached as Exhibit 99.1", ["99.1"]),
        ("Exhibit 99.2 and Exhibit 99.3", ["99.2", "99.3"]),
        ("furnished as Exhibit 99", ["99"]),
        ("EX-99.1", ["99.1"]),
        ("ex 99.1", ["99.1"]),
        ("Exhibit 10.1", []),
        ("Exhibit 101.SCH", []),
    ],
)
def test_pattern_matches_real_reference_forms(exhibit_re, text, expected):
    assert exhibit_re.findall(text) == expected


# --------------------------------------------------------------------------
# Label resolution
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "label, present, expected",
    [
        ("99.1", {"99.1"}, True),
        ("99.1", {"99.1", "99.2"}, True),
        ("99.2", {"99.1"}, False),
        # Prose saying "Exhibit 99" against a filing carrying EX-99.1: the same
        # document, so requiring an exact string match would invent a gap.
        ("99", {"99.1"}, True),
        ("99", {"99.2", "99.3"}, True),
        ("99", {"99"}, True),
        ("99", set(), False),
        ("99.1", set(), False),
        # A sub-numbered label must NOT be satisfied by a different sub-number.
        ("99.4", {"99.1", "99.2", "99.3"}, False),
    ],
)
def test_is_resolved(label, present, expected):
    assert _is_resolved(label, present) is expected


# --------------------------------------------------------------------------
# The item / cover-page split
# --------------------------------------------------------------------------


def test_item_split_separates_cover_page_from_narrative():
    item_re = re.compile(load_config().get("audit.item_pattern"), re.I)
    sig_re = re.compile(load_config().get("audit.signature_pattern"), re.I)
    text = (
        "ALBERTSONS COMPANIES INC Commission File Number 001-39350 "
        "Item 2.02 Results of Operations and Financial Condition. "
        "On September 8 2026 the company issued a press release. "
        "SIGNATURES Pursuant to the requirements of the Securities Exchange Act"
    )
    chars, items = _split_item_narrative(text, item_re, sig_re)
    assert items == ["2.02"]
    assert 0 < chars < len(text), "the split must exclude cover page and signature"
    assert "Commission File Number" not in text[text.index("Item 2.02") :][:chars]


def test_item_split_reports_zero_when_the_pattern_does_not_apply():
    """A 10-Q has no N.NN items; that must read as 'not applicable', not 'empty'."""
    item_re = re.compile(load_config().get("audit.item_pattern"), re.I)
    sig_re = re.compile(load_config().get("audit.signature_pattern"), re.I)
    chars, items = _split_item_narrative(
        "Item 2. Management's Discussion and Analysis of Financial Condition",
        item_re,
        sig_re,
    )
    assert (chars, items) == (0, [])


def test_item_split_runs_to_end_when_there_is_no_signature_block():
    item_re = re.compile(load_config().get("audit.item_pattern"), re.I)
    sig_re = re.compile(load_config().get("audit.signature_pattern"), re.I)
    chars, _ = _split_item_narrative("Item 8.01 Other Events. Something.", item_re, sig_re)
    assert chars == len("Item 8.01 Other Events. Something.")
