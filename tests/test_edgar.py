"""Tests for EDGAR identifiers, the index-headers parser, and CSV schemas."""

from __future__ import annotations

import pytest

from tools.edgar import (
    DOC_TYPE_EXHIBIT,
    DOC_TYPE_PRIMARY,
    LEGACY_MANIFEST_FIELDS,
    MANIFEST_FIELDS,
    EdgarError,
    RateLimiter,
    accession_dashed,
    accession_nodash,
    cik10,
    filing_index_url,
    iter_recent_filings,
    matches_exhibit_type,
    parse_index_headers,
    primary_doc_url,
    read_manifest,
    stored_filename,
    write_manifest,
)

ARCHIVES = "https://www.sec.gov/Archives"


# --------------------------------------------------------------------------
# Identifiers and URLs
# --------------------------------------------------------------------------


def test_cik_and_accession_forms():
    assert cik10(320193) == "0000320193"
    assert cik10("CIK0000320193") == "0000320193"
    assert accession_dashed("000032019324000081") == "0000320193-24-000081"
    assert accession_nodash("0000320193-24-000081") == "000032019324000081"


@pytest.mark.parametrize("bad", ["not-a-cik", "", "12ab"])
def test_bad_cik_rejected(bad):
    with pytest.raises(EdgarError):
        cik10(bad)


def test_short_accession_rejected():
    with pytest.raises(EdgarError):
        accession_dashed("123")


def test_archives_urls_unpad_the_cik():
    """data.sec.gov pads the CIK; the Archives path does not."""
    assert primary_doc_url(ARCHIVES, "0001646972", "0001646972-26-000054", "aci.htm") == (
        "https://www.sec.gov/Archives/edgar/data/1646972/000164697226000054/aci.htm"
    )
    assert filing_index_url(ARCHIVES, "0001646972", "0001646972-26-000054") == (
        "https://www.sec.gov/Archives/edgar/data/1646972/000164697226000054/"
        "0001646972-26-000054-index-headers.html"
    )


# --------------------------------------------------------------------------
# index-headers parsing -- the authoritative source of document types
# --------------------------------------------------------------------------

#: Shape observed in a real filing: escaped SGML inside a <PRE> block, carrying a
#: primary document, an EX-99.1, and EX-101.* XBRL taxonomy files.
INDEX_HEADERS = """<HTML><HEAD><TITLE>SEC EDGAR Submission</TITLE></HEAD><BODY><PRE>
&lt;SEC-HEADER&gt;0001646972-26-000054.hdr.sgml : 20260909
&lt;ACCESSION-NUMBER&gt;0001646972-26-000054
&lt;/SEC-HEADER&gt;
&lt;DOCUMENT&gt;
&lt;TYPE&gt;8-K
&lt;SEQUENCE&gt;1
&lt;FILENAME&gt;aci-20260908.htm
&lt;DESCRIPTION&gt;8-K
&lt;TEXT&gt;
&lt;/DOCUMENT&gt;
&lt;DOCUMENT&gt;
&lt;TYPE&gt;EX-99.1
&lt;SEQUENCE&gt;2
&lt;FILENAME&gt;ex991-pressreleasebod.htm
&lt;DESCRIPTION&gt;EX-99.1
&lt;TEXT&gt;
&lt;/DOCUMENT&gt;
&lt;DOCUMENT&gt;
&lt;TYPE&gt;EX-101.SCH
&lt;SEQUENCE&gt;3
&lt;FILENAME&gt;aci-20260908.xsd
&lt;DESCRIPTION&gt;XBRL TAXONOMY EXTENSION SCHEMA DOCUMENT
&lt;TEXT&gt;
&lt;/DOCUMENT&gt;
</PRE></BODY></HTML>"""


def test_parse_index_headers_extracts_types_and_filenames():
    docs = parse_index_headers(INDEX_HEADERS)
    assert [d["type"] for d in docs] == ["8-K", "EX-99.1", "EX-101.SCH"]
    assert docs[1]["filename"] == "ex991-pressreleasebod.htm"
    assert docs[1]["sequence"] == "2"
    assert docs[1]["description"] == "EX-99.1"


def test_exhibit_prefix_excludes_xbrl_taxonomy():
    """EX-101.* sits in the same index and must not be mistaken for an exhibit."""
    docs = parse_index_headers(INDEX_HEADERS)
    matched = [d for d in docs if matches_exhibit_type(d["type"], ["EX-99"])]
    assert [d["type"] for d in matched] == ["EX-99.1"]


@pytest.mark.parametrize(
    "doc_type, prefixes, expected",
    [
        ("EX-99", ["EX-99"], True),
        ("EX-99.1", ["EX-99"], True),
        ("ex-99.2", ["EX-99"], True),      # case-insensitive
        ("EX-101.SCH", ["EX-99"], False),
        ("EX-10.1", ["EX-99"], False),
        ("8-K", ["EX-99"], False),
        (None, ["EX-99"], False),
        ("EX-99.1", [], False),
    ],
)
def test_matches_exhibit_type(doc_type, prefixes, expected):
    assert matches_exhibit_type(doc_type, prefixes) is expected


def test_document_block_without_filename_is_reported_not_invented():
    payload = (
        "<PRE>&lt;DOCUMENT&gt;\n&lt;TYPE&gt;EX-99.1\n&lt;SEQUENCE&gt;2\n"
        "&lt;TEXT&gt;\n&lt;/DOCUMENT&gt;</PRE>"
    )
    docs = parse_index_headers(payload)
    assert len(docs) == 1
    assert docs[0]["type"] == "EX-99.1"
    assert docs[0]["filename"] is None


def test_parse_index_headers_on_empty_payload():
    assert parse_index_headers("") == []


# --------------------------------------------------------------------------
# On-disk naming
# --------------------------------------------------------------------------


def test_primary_filename_is_unchanged_by_the_exhibit_schema():
    """This is what makes extending the corpus cost zero re-fetching."""
    assert (
        stored_filename("0001646972-26-000054", DOC_TYPE_PRIMARY, "aci-20260908.htm", ".html")
        == "000164697226000054.html"
    )


def test_exhibit_filename_keeps_the_source_document_stem():
    assert stored_filename(
        "0001646972-26-000054", DOC_TYPE_EXHIBIT, "ex991-pressreleasebod.htm", ".html"
    ) == "000164697226000054-ex991-pressreleasebod.html"


def test_exhibit_filename_sanitizes_unsafe_characters():
    name = stored_filename(
        "0001646972-26-000054", DOC_TYPE_EXHIBIT, "ex 99 press:release.htm", ".html"
    )
    assert name == "000164697226000054-ex_99_press_release.html"
    assert " " not in name and ":" not in name


@pytest.mark.parametrize(
    "document",
    ["../../../etc/passwd", "sub/dir/ex991.htm", r"..\..\windows\system32"],
)
def test_exhibit_filename_cannot_escape_the_output_directory(document):
    """A document name is attacker-adjacent data: it comes from a fetched file."""
    name = stored_filename("0001646972-26-000054", DOC_TYPE_EXHIBIT, document, ".html")
    assert "/" not in name and "\\" not in name
    assert ".." not in name
    assert name.startswith("000164697226000054-")


def test_two_exhibits_of_the_same_label_get_distinct_names():
    """Naming by label rather than filename would collide here."""
    a = stored_filename("0001-26-000001", DOC_TYPE_EXHIBIT, "ex991-a.htm", ".html")
    b = stored_filename("0001-26-000001", DOC_TYPE_EXHIBIT, "ex991-b.htm", ".html")
    assert a != b


# --------------------------------------------------------------------------
# Manifest schema and determinism
# --------------------------------------------------------------------------


def _row(**kw):
    base = {
        "cik": "0000000001",
        "ticker": "TEST",
        "accession": "0000000001-26-000001",
        "form": "8-K",
        "filing_date": "2026-01-02",
        "period_of_report": "2026-01-01",
        "doc_type": DOC_TYPE_PRIMARY,
        "exhibit_label": None,
        "document": "primary.htm",
        "doc_url": "https://example.test/primary.htm",
    }
    base.update(kw)
    return base


def test_manifest_is_byte_identical_regardless_of_input_order(tmp_path):
    rows = [
        _row(),
        _row(doc_type=DOC_TYPE_EXHIBIT, exhibit_label="EX-99.2", document="b.htm"),
        _row(doc_type=DOC_TYPE_EXHIBIT, exhibit_label="EX-99.1", document="a.htm"),
        _row(accession="0000000001-26-000002", filing_date="2026-03-04", document="p2.htm"),
    ]
    path = tmp_path / "manifest.csv"
    write_manifest(path, rows)
    first = path.read_bytes()
    write_manifest(path, list(reversed(rows)))
    assert path.read_bytes() == first


def test_manifest_sorts_primary_before_its_exhibits(tmp_path):
    path = tmp_path / "manifest.csv"
    write_manifest(
        path,
        [
            _row(doc_type=DOC_TYPE_EXHIBIT, exhibit_label="EX-99.1", document="a.htm"),
            _row(),
        ],
    )
    back = read_manifest(path)
    assert [r["doc_type"] for r in back] == [DOC_TYPE_PRIMARY, DOC_TYPE_EXHIBIT]


def test_none_is_written_blank_not_as_the_string_none(tmp_path):
    path = tmp_path / "manifest.csv"
    write_manifest(path, [_row(period_of_report=None, doc_url=None)])
    back = read_manifest(path)
    assert back[0]["period_of_report"] == ""
    assert back[0]["doc_url"] == ""
    assert "None" not in path.read_text(encoding="utf-8")


def test_legacy_manifest_is_refused_by_default_and_names_the_fix(tmp_path):
    path = tmp_path / "manifest.csv"
    path.write_text(
        ",".join(LEGACY_MANIFEST_FIELDS)
        + "\n0000000001,TEST,0000000001-26-000001,8-K,2026-01-02,2026-01-01,"
        "https://example.test/edgar/data/1/x/primary.htm\n",
        encoding="utf-8",
    )
    with pytest.raises(EdgarError, match="tools.discover"):
        read_manifest(path)


def test_legacy_manifest_upgrade_sources_the_document_from_the_url(tmp_path):
    """The basename of primary_doc_url is EDGAR's filename, not a guess."""
    path = tmp_path / "manifest.csv"
    path.write_text(
        ",".join(LEGACY_MANIFEST_FIELDS)
        + "\n0000000001,TEST,0000000001-26-000001,8-K,2026-01-02,2026-01-01,"
        "https://example.test/edgar/data/1/x/aci-20260908.htm\n",
        encoding="utf-8",
    )
    rows = read_manifest(path, allow_legacy=True)
    assert tuple(rows[0]) == MANIFEST_FIELDS
    assert rows[0]["doc_type"] == DOC_TYPE_PRIMARY
    assert rows[0]["document"] == "aci-20260908.htm"
    assert rows[0]["exhibit_label"] == ""


# --------------------------------------------------------------------------
# Columnar submissions data
# --------------------------------------------------------------------------


def test_short_column_truncates_rather_than_mispairing():
    """A short array must not pair a form with another filing's date."""
    subs = {
        "filings": {
            "recent": {
                "accessionNumber": ["a", "b", "c"],
                "filingDate": ["2024-05-01", "2024-08-01", "2025-02-01"],
                "reportDate": ["2024-03-31", "2024-06-30"],  # short
                "form": ["10-Q", "8-K", "10-Q"],
            }
        }
    }
    rows = list(iter_recent_filings(subs))
    assert len(rows) == 2
    assert rows[1] == {
        "accessionNumber": "b",
        "filingDate": "2024-08-01",
        "reportDate": "2024-06-30",
        "form": "8-K",
    }


# --------------------------------------------------------------------------
# Rate limiting -- SEC's ceiling is per requester, not per connection
# --------------------------------------------------------------------------


def test_concurrency_cannot_raise_the_request_rate():
    import threading
    import time

    limiter = RateLimiter(50.0)  # 20ms apart
    workers, per_worker = 8, 4
    total = workers * per_worker

    def run():
        for _ in range(per_worker):
            limiter.acquire()

    threads = [threading.Thread(target=run) for _ in range(workers)]
    start = time.monotonic()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.monotonic() - start

    floor = (total - 1) / 50.0
    assert elapsed >= floor * 0.95, (
        f"{total} acquisitions across {workers} threads took {elapsed:.3f}s, "
        f"below the {floor:.3f}s the configured rate requires"
    )


def test_penalize_delays_the_whole_pool():
    import time

    limiter = RateLimiter(1000.0)
    limiter.acquire()
    limiter.penalize(0.25)
    start = time.monotonic()
    limiter.acquire()
    assert time.monotonic() - start >= 0.2


def test_non_positive_rate_is_rejected():
    with pytest.raises(EdgarError):
        RateLimiter(0)
