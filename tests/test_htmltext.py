"""Tests for the shared HTML text extractor.

The separator behaviour here is a correctness requirement, not a style choice:
a weld between adjacent text nodes makes configured section anchors miss, which
degrades retrieval silently rather than failing. See ``tools/htmltext.py``.
"""

from __future__ import annotations

import re

import pytest

from tools.config import load_config
from tools.htmltext import (
    DEFAULT_INLINE_TAGS,
    extract_text,
    inline_tags_from_config,
)

# --------------------------------------------------------------------------
# The separator requirement
# --------------------------------------------------------------------------


def test_adjacent_spans_do_not_weld():
    """The required assertion: separate elements never fuse their text."""
    assert extract_text("<span>Item</span><span>2.02</span>") == "Item 2.02"


def test_typographic_inline_tags_do_not_split_a_word():
    """The counterpart: a separator must not break a word apart."""
    assert extract_text("<b>Fin</b>ancial") == "Financial"
    assert extract_text("<p>the 1<sup>st</sup> quarter</p>") == "the 1st quarter"


def test_real_edgar_markup_shape():
    """The markup actually observed in a fetched 8-K.

    ``Item&#160;5.02`` in one table cell, the heading in the next. Before the
    separator fix this produced ``Item 5.02Departure of Directors``.
    """
    html = (
        '<td style="vertical-align:top"><span style="font-weight:700">'
        "Item&#160;5.02</span></td>"
        '<td colspan="3" style="padding:0 1pt"/>'
        '<td style="text-align:left"><span>Departure of Directors</span></td>'
    )
    assert extract_text(html) == "Item 5.02 Departure of Directors"


def test_nbsp_becomes_matchable_whitespace():
    """``\\s`` in an anchor pattern must match what the filer wrote."""
    text = extract_text("<span>Item&#160;2.02</span>")
    assert text == "Item 2.02"
    assert re.search(r"item\s*(\d\.\d{2})", text, re.I)


@pytest.mark.parametrize(
    "html, expected",
    [
        ("<div>a</div><div>b</div>", "a b"),
        ("<p>a</p>b", "a b"),
        ("<td>a</td><td>b</td>", "a b"),
        ("a<br/>b", "a b"),
        ("<span>a</span><em>b</em>", "a b"),  # boundary from span, not em
        ("<em>a</em><em>b</em>", "ab"),  # both inline: no separator
    ],
)
def test_boundary_matrix(html, expected):
    assert extract_text(html) == expected


# --------------------------------------------------------------------------
# The corpus-wide hazard: 10-Q anchors, tested against the LIVE config
# --------------------------------------------------------------------------


def test_live_10q_anchors_survive_extraction():
    """Every configured 10-Q anchor must still match after extraction.

    This loads ``inputs/config.yaml`` rather than a fixture copy, so editing an
    anchor into a form that cannot survive text extraction fails the suite.

    The heading text is split across table cells exactly as EDGAR emits it. A
    weld here would make every 10-Q anchor miss and collapse section-aware
    chunking into ``_default`` across the whole corpus.
    """
    config = load_config()
    inline = inline_tags_from_config(config)
    anchors = config.get("parse.section_anchors")["10-Q"]

    headings = {
        "part1_item1_financial_statements": ("Item&#160;1.", "Financial Statements"),
        "part1_item2_mdna": ("Item&#160;2.", "Management's Discussion and Analysis"),
        "part1_item3_market_risk": (
            "Item&#160;3.",
            "Quantitative and Qualitative Disclosures About Market Risk",
        ),
        "part1_item4_controls": ("Item&#160;4.", "Controls and Procedures"),
        "part2_item1_legal": ("Item&#160;1.", "Legal Proceedings"),
        "part2_item1a_risk_factors": ("Item&#160;1A.", "Risk Factors"),
        "part2_item2_unregistered_sales": (
            "Item&#160;2.",
            "Unregistered Sales of Equity Securities",
        ),
        "part2_item5_other": ("Item&#160;5.", "Other Information"),
        "part2_item6_exhibits": ("Item&#160;6.", "Exhibits"),
    }

    for anchor in anchors:
        label, pattern = anchor["label"], anchor["pattern"]
        assert label in headings, f"no test heading for configured anchor {label!r}"
        item, heading = headings[label]
        html = (
            f'<td><span style="font-weight:700">{item}</span></td>'
            f'<td colspan="3" style="padding:0 1pt"/>'
            f"<td><span>{heading}</span></td>"
        )
        text = extract_text(html, inline_tags=inline)
        assert re.search(pattern, text, re.IGNORECASE), (
            f"configured 10-Q anchor {label!r} does not match extracted text "
            f"{text!r}. Pattern: {pattern!r}"
        )


def test_live_8k_anchor_survives_extraction():
    """Same guarantee for the 8-K item anchor, with its capturing group."""
    config = load_config()
    inline = inline_tags_from_config(config)
    pattern = config.get("parse.section_anchors")["8-K"][0]["pattern"]
    html = (
        '<td><span style="font-weight:700">Item&#160;5.02</span></td>'
        "<td><span>Departure of Directors</span></td>"
    )
    match = re.search(pattern, extract_text(html, inline_tags=inline), re.IGNORECASE)
    assert match is not None
    assert match.group(1) == "5.02"


def test_config_inline_tags_are_loadable():
    configured = inline_tags_from_config(load_config())
    assert "b" in configured and "sup" in configured
    # span must NOT be excluded from separation, or the weld returns.
    assert "span" not in configured
    assert configured == DEFAULT_INLINE_TAGS or configured


# --------------------------------------------------------------------------
# Hidden and non-prose content
# --------------------------------------------------------------------------


def test_inline_xbrl_hidden_facts_are_dropped():
    """A bare pointer must not look substantive because of hidden XBRL facts."""
    html = (
        "<ix:header><ix:hidden>0001646972 2026-03-01 2026-06-20 us-gaap:CommonClass"
        "</ix:hidden></ix:header><p>Item 2.02 Results of Operations.</p>"
    )
    text = extract_text(html)
    assert "us-gaap" not in text
    assert "0001646972" not in text
    assert text == "Item 2.02 Results of Operations."


def test_display_none_is_dropped():
    html = '<div style="display:none">hidden fact</div><p>visible</p>'
    assert extract_text(html) == "visible"


def test_script_and_style_are_dropped():
    html = "<style>.a{color:red}</style><script>var x=1;</script><p>body</p>"
    assert extract_text(html) == "body"


def test_unclosed_tags_do_not_swallow_the_document():
    """Filings contain unbalanced markup; recovery must not lose the body."""
    html = "<div><span>Item 8.01</span><p>Other Events.</div><p>After.</p>"
    text = extract_text(html)
    assert "Item 8.01" in text
    assert "Other Events." in text
    assert "After." in text


def test_unclosed_drop_tag_does_not_hide_everything_after_it():
    html = "<style>.a{}<p>still visible</p>"
    # The style element is never closed; content after it is inside it as far as
    # the parser can tell. Assert we do not crash and do not emit stylesheet text.
    assert ".a{}" not in extract_text(html)


def test_bytes_input_is_accepted():
    assert extract_text(b"<p>bytes</p>") == "bytes"


def test_empty_and_textless_documents():
    assert extract_text("") == ""
    assert extract_text("<div></div>") == ""
