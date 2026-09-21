"""Stage-level tests for discover and fetch, against stub clients.

No network. The stub records every URL it is asked for, so "did this cost a
request?" is directly assertable -- which is the property that matters for the
skip-before-request guarantee and for the exhibit index cost.
"""

from __future__ import annotations

import csv
import datetime as dt
from pathlib import Path

import pytest

from tools.config import load_config
from tools.discover import _discover_one
from tools.edgar import (
    DOC_TYPE_EXHIBIT,
    DOC_TYPE_PRIMARY,
    LEDGER_FIELDS,
    LEGACY_LEDGER_FIELDS,
    MANIFEST_FIELDS,
    EdgarError,
    manifest_key,
)
from tools.fetch import _already_fetched, _load_ledger
from tools.runlog import RunLog

SUBS = "https://data.sec.gov/submissions"
ARCHIVES = "https://www.sec.gov/Archives"
CIK = "0000000001"
ENTRY = {"cik": CIK, "ticker": "TEST", "company_name": "Test Co"}
ACC = [
    "0000000001-25-000009",
    "0000000001-25-000008",
    "0000000001-24-000007",
    "0000000001-24-000006",
]


def _index_headers(*docs: tuple[str, str | None]) -> str:
    """Build an index-headers payload from (type, filename) pairs."""
    blocks = []
    for i, (doc_type, filename) in enumerate(docs, start=1):
        block = f"&lt;DOCUMENT&gt;\n&lt;TYPE&gt;{doc_type}\n&lt;SEQUENCE&gt;{i}\n"
        if filename is not None:
            block += f"&lt;FILENAME&gt;{filename}\n"
        block += "&lt;TEXT&gt;\n&lt;/DOCUMENT&gt;"
        blocks.append(block)
    return "<HTML><BODY><PRE>\n" + "\n".join(blocks) + "\n</PRE></BODY></HTML>"


class _Response:
    def __init__(self, text: str) -> None:
        self.text = text


class StubClient:
    """Stands in for EdgarClient, recording every URL requested."""

    def __init__(self, docs: dict[str, object]) -> None:
        self.docs = docs
        self.calls: list[str] = []

    def get_json(self, url: str):
        self.calls.append(url)
        if url not in self.docs:
            raise EdgarError(f"stub has no {url}")
        return self.docs[url]

    def get(self, url: str) -> _Response:
        self.calls.append(url)
        if url not in self.docs:
            raise EdgarError(f"stub has no {url}")
        return _Response(str(self.docs[url]))


SUBMISSIONS = {
    "filings": {
        "recent": {
            "accessionNumber": ACC,
            "filingDate": ["2025-05-01", "2025-02-10", "2024-08-01", "2024-03-01"],
            "reportDate": ["2025-03-31", "2024-12-31", "2024-06-30", ""],
            "form": ["10-Q", "10-Q/A", "8-K", "8-K"],
            "primaryDocument": ["q1.htm", "amend.htm", "ev.htm", "ev2.htm"],
        },
        "files": [],
    }
}


def _index_url(accession: str) -> str:
    from tools.edgar import filing_index_url

    return filing_index_url(ARCHIVES, CIK, accession)


@pytest.fixture
def stage(tmp_path):
    config = load_config()
    with RunLog(config, run_id="test", log_dir=str(tmp_path)) as log:
        with log.stage("discover") as rec:
            yield rec


@pytest.fixture
def config():
    return load_config()


# --------------------------------------------------------------------------
# discover: exhibit expansion
# --------------------------------------------------------------------------


def _client_with_exhibits() -> StubClient:
    return StubClient(
        {
            f"{SUBS}/CIK{CIK}.json": SUBMISSIONS,
            # 8-K with a qualifying exhibit plus XBRL taxonomy noise
            _index_url(ACC[2]): _index_headers(
                ("8-K", "ev.htm"),
                ("EX-99.1", "ex991-press.htm"),
                ("EX-101.SCH", "tick.xsd"),
            ),
            # 8-K with no qualifying exhibit
            _index_url(ACC[3]): _index_headers(
                ("8-K", "ev2.htm"), ("EX-101.SCH", "tick.xsd")
            ),
        }
    )


def test_exhibit_rows_are_added_for_qualifying_types(config, stage):
    client = _client_with_exhibits()
    result = _discover_one(
        ENTRY, client, config, stage, (dt.date(2024, 1, 1), dt.date(2025, 12, 31))
    )
    exhibits = [r for r in result["rows"] if r["doc_type"] == DOC_TYPE_EXHIBIT]
    assert len(exhibits) == 1
    row = exhibits[0]
    assert row["exhibit_label"] == "EX-99.1"
    assert row["document"] == "ex991-press.htm"
    assert row["doc_url"].endswith("/000000000124000007/ex991-press.htm")
    # Provenance is carried from the parent filing, not re-derived.
    assert row["accession"] == ACC[2]
    assert row["form"] == "8-K"
    assert row["filing_date"] == "2024-08-01"
    assert row["period_of_report"] == "2024-06-30"


def test_xbrl_taxonomy_documents_never_become_rows(config, stage):
    client = _client_with_exhibits()
    result = _discover_one(
        ENTRY, client, config, stage, (dt.date(2024, 1, 1), dt.date(2025, 12, 31))
    )
    assert all("EX-101" not in (r["exhibit_label"] or "") for r in result["rows"])
    assert all(not (r["document"] or "").endswith(".xsd") for r in result["rows"])


def test_tripwire_counts_split_with_and_without_exhibits(config, stage):
    """The count that must come back near 70 on the real corpus."""
    client = _client_with_exhibits()
    result = _discover_one(
        ENTRY, client, config, stage, (dt.date(2024, 1, 1), dt.date(2025, 12, 31))
    )
    assert result["indexed"] == 2            # both 8-Ks indexed
    assert result["with_exhibit"] == 1
    assert result["without_exhibit"] == 1
    assert result["index_failed"] == 0
    assert result["types_seen"]["EX-101.SCH"] == 2


def test_only_configured_forms_are_indexed(config, stage):
    """10-Q filings cost no index request under the default exhibit_forms."""
    client = _client_with_exhibits()
    _discover_one(
        ENTRY, client, config, stage, (dt.date(2024, 1, 1), dt.date(2025, 12, 31))
    )
    index_calls = [c for c in client.calls if "index-headers" in c]
    assert len(index_calls) == 2
    assert all(f"000000000124" in c for c in index_calls)  # the two 8-Ks only


def test_index_failure_is_counted_apart_from_having_no_exhibit(config, stage):
    """A failed request must not masquerade as a structural fact."""
    client = StubClient(
        {
            f"{SUBS}/CIK{CIK}.json": SUBMISSIONS,
            _index_url(ACC[2]): _index_headers(
                ("8-K", "ev.htm"), ("EX-99.1", "ex991-press.htm")
            ),
            # ACC[3]'s index is absent -> the stub raises
        }
    )
    result = _discover_one(
        ENTRY, client, config, stage, (dt.date(2024, 1, 1), dt.date(2025, 12, 31))
    )
    assert result["index_failed"] == 1
    assert result["without_exhibit"] == 0, (
        "a failed index request was counted as 'no qualifying exhibit', which "
        "would corrupt the tripwire"
    )
    assert result["with_exhibit"] == 1
    # The filing keeps its primary row despite the index failure.
    assert any(
        r["accession"] == ACC[3] and r["doc_type"] == DOC_TYPE_PRIMARY
        for r in result["rows"]
    )


def test_exhibit_without_filename_is_skipped_and_logged(config, stage):
    client = StubClient(
        {
            f"{SUBS}/CIK{CIK}.json": SUBMISSIONS,
            _index_url(ACC[2]): _index_headers(("8-K", "ev.htm"), ("EX-99.1", None)),
            _index_url(ACC[3]): _index_headers(("8-K", "ev2.htm")),
        }
    )
    before = stage.errors
    result = _discover_one(
        ENTRY, client, config, stage, (dt.date(2024, 1, 1), dt.date(2025, 12, 31))
    )
    assert not [r for r in result["rows"] if r["doc_type"] == DOC_TYPE_EXHIBIT]
    assert stage.errors > before, "a dropped exhibit must be logged, not silent"


def test_exhibit_types_are_configurable(stage):
    """Rule 3: widening the corpus is config, not a code change."""
    config = load_config(overrides=['discover.exhibit_types=["EX-101"]'])
    client = _client_with_exhibits()
    result = _discover_one(
        ENTRY, client, config, stage, (dt.date(2024, 1, 1), dt.date(2025, 12, 31))
    )
    labels = {r["exhibit_label"] for r in result["rows"] if r["doc_type"] == DOC_TYPE_EXHIBIT}
    assert labels == {"EX-101.SCH"}


def test_primary_rows_still_carry_the_window_and_form_rules(config, stage):
    client = _client_with_exhibits()
    result = _discover_one(
        ENTRY, client, config, stage, (dt.date(2024, 1, 1), dt.date(2025, 12, 31))
    )
    primaries = {r["accession"] for r in result["rows"] if r["doc_type"] == DOC_TYPE_PRIMARY}
    assert ACC[1] not in primaries, "10-Q/A must not match the exact form list"
    assert primaries == {ACC[0], ACC[2], ACC[3]}


# --------------------------------------------------------------------------
# fetch: ledger migration and the skip key
# --------------------------------------------------------------------------


def _write_legacy_ledger(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(LEGACY_LEDGER_FIELDS),
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def test_legacy_ledger_migration_keeps_every_document_skippable(tmp_path):
    """The additive guarantee: nothing already fetched is re-fetched."""
    doc = tmp_path / "000000000125000009.html"
    doc.write_bytes(b"<html>primary</html>")
    ledger_path = tmp_path / "fetched.csv"
    _write_legacy_ledger(
        ledger_path,
        [
            {
                "accession": ACC[0],
                "path": str(doc),
                "status": "ok",
                "bytes": str(doc.stat().st_size),
                "content_type": "text/html",
                "sha256": "x",
                "fetched_at": "now",
            }
        ],
    )
    manifest = [
        {
            **{k: "" for k in MANIFEST_FIELDS},
            "accession": ACC[0],
            "doc_type": DOC_TYPE_PRIMARY,
            "document": "q1.htm",
        }
    ]
    ledger, migrated = _load_ledger(ledger_path, manifest)

    assert migrated == 1
    # Keyed by the manifest's document name, so the new-schema skip check hits.
    assert (ACC[0], "q1.htm") in ledger
    assert ledger[(ACC[0], "q1.htm")]["doc_type"] == DOC_TYPE_PRIMARY
    assert _already_fetched(manifest_key(manifest[0]), ledger) is not None


def test_legacy_row_with_no_matching_manifest_row_is_left_blank_not_guessed(tmp_path):
    doc = tmp_path / "orphan.html"
    doc.write_bytes(b"x")
    ledger_path = tmp_path / "fetched.csv"
    _write_legacy_ledger(
        ledger_path,
        [
            {
                "accession": "0000000009-99-999999",
                "path": str(doc),
                "status": "ok",
                "bytes": "1",
                "content_type": "text/html",
                "sha256": "x",
                "fetched_at": "now",
            }
        ],
    )
    ledger, migrated = _load_ledger(ledger_path, manifest=[])
    assert migrated == 1
    assert ("0000000009-99-999999", "") in ledger
    assert ledger[("0000000009-99-999999", "")]["document"] == ""


def test_exhibit_and_primary_of_one_filing_are_tracked_separately(tmp_path):
    """Accession alone is no longer identity."""
    primary = tmp_path / "p.html"
    primary.write_bytes(b"primary")
    ledger_path = tmp_path / "fetched.csv"
    with ledger_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(LEDGER_FIELDS), lineterminator="\n")
        writer.writeheader()
        writer.writerow(
            {
                "accession": ACC[2],
                "document": "ev.htm",
                "doc_type": DOC_TYPE_PRIMARY,
                "path": str(primary),
                "status": "ok",
                "bytes": str(primary.stat().st_size),
                "content_type": "text/html",
                "sha256": "x",
                "fetched_at": "now",
            }
        )
    ledger, migrated = _load_ledger(ledger_path, manifest=[])
    assert migrated == 0
    assert _already_fetched((ACC[2], "ev.htm"), ledger) is not None
    assert _already_fetched((ACC[2], "ex991-press.htm"), ledger) is None, (
        "an unfetched exhibit must not be skipped because its filing's primary "
        "document is present"
    )


@pytest.mark.parametrize("damage", ["truncate", "delete"])
def test_damaged_file_is_not_skipped(tmp_path, damage):
    doc = tmp_path / "d.html"
    doc.write_bytes(b"<html>full content</html>")
    ledger_path = tmp_path / "fetched.csv"
    _write_legacy_ledger(
        ledger_path,
        [
            {
                "accession": ACC[0],
                "path": str(doc),
                "status": "ok",
                "bytes": str(doc.stat().st_size),
                "content_type": "text/html",
                "sha256": "x",
                "fetched_at": "now",
            }
        ],
    )
    manifest = [
        {
            **{k: "" for k in MANIFEST_FIELDS},
            "accession": ACC[0],
            "doc_type": DOC_TYPE_PRIMARY,
            "document": "q1.htm",
        }
    ]
    ledger, _ = _load_ledger(ledger_path, manifest)
    assert _already_fetched((ACC[0], "q1.htm"), ledger) is not None

    if damage == "truncate":
        doc.write_bytes(b"short")
    else:
        doc.unlink()
    assert _already_fetched((ACC[0], "q1.htm"), ledger) is None


def test_error_status_is_retried_next_run(tmp_path):
    ledger_path = tmp_path / "fetched.csv"
    _write_legacy_ledger(
        ledger_path,
        [
            {
                "accession": ACC[0],
                "path": "",
                "status": "error",
                "bytes": "0",
                "content_type": "",
                "sha256": "",
                "fetched_at": "now",
            }
        ],
    )
    ledger, _ = _load_ledger(ledger_path, manifest=[])
    assert _already_fetched((ACC[0], ""), ledger) is None


def test_unknown_ledger_header_is_rejected(tmp_path):
    ledger_path = tmp_path / "fetched.csv"
    ledger_path.write_text("a,b,c\n1,2,3\n", encoding="utf-8")
    with pytest.raises(EdgarError, match="header"):
        _load_ledger(ledger_path, manifest=[])
