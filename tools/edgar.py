"""Shared SEC EDGAR access: rate limiting, retries, URL construction, CSV schemas.

Three separate things in this harness talk to SEC over HTTP -- the universe
build, the discover stage, and the fetch stage -- and all three owe SEC the same
manners. They share this module so those manners cannot drift apart between
stages.

THE RATE CEILING IS PER REQUESTER, NOT PER CONNECTION
=====================================================

SEC publishes a ceiling of 10 requests per second, and states it applies
"regardless of the number of machines used to submit requests"
(https://www.sec.gov/about/privacy-information#security). The Webmaster FAQ
repeats it: "our current maximum access rate is 10 requests per second. This is
carefully monitored to preserve equitable access for all users."
(https://www.sec.gov/os/webmaster-faq)

Two consequences are built into this module rather than left to each caller:

1. The limiter is ONE object shared by every worker thread of a client. Raising
   ``fetch.concurrency`` therefore cannot raise the achieved request rate -- it
   only changes how many requests are in flight while waiting on the same
   budget. Concurrency and rate stay independent knobs, which is what makes a
   throughput sweep interpretable.

2. Overshooting does not cost one failed call. SEC limits the offending IP and
   the caller may resume only "once the rate of requests has dropped below the
   threshold for 10 minutes", and SEC reserves "the right to block IP addresses
   that submit excessive requests." So backoff must be able to run well past a
   single retry interval, and a 429 is treated as a hard signal to slow down,
   not as a transient blip.

SEC also requires a declared identity: "Please declare your user agent in
request headers". The client refuses to construct without one rather than
sending an anonymous request that SEC would answer with 403.

INSTRUMENTATION
===============

Every HTTP attempt calls ``stage.request()`` -- retries included, because a
retry is real load on SEC and a run log that hid them would understate the cost
of a flaky run. Response bodies call ``stage.bytes_in()``. Callers add their own
``count()`` and ``bytes_out()``.
"""

from __future__ import annotations

import csv
import html
import random
import re
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Iterator

if TYPE_CHECKING:  # pragma: no cover
    import httpx

    from tools.config import Config
    from tools.runlog import StageRecorder

__all__ = [
    "EdgarError",
    "RateLimiter",
    "EdgarClient",
    "accession_nodash",
    "accession_dashed",
    "cik10",
    "primary_doc_url",
    "filing_index_url",
    "parse_index_headers",
    "matches_exhibit_type",
    "stored_filename",
    "as_stored_path",
    "from_stored_path",
    "UNIVERSE_FIELDS",
    "MANIFEST_FIELDS",
    "LEGACY_MANIFEST_FIELDS",
    "LEDGER_FIELDS",
    "LEGACY_LEDGER_FIELDS",
    "DOC_TYPE_PRIMARY",
    "DOC_TYPE_EXHIBIT",
    "load_universe",
    "write_universe",
    "read_manifest",
    "write_manifest",
    "manifest_key",
]


class EdgarError(RuntimeError):
    """A request to SEC failed, or a response could not be used as given."""


# --------------------------------------------------------------------------
# Rate limiting
# --------------------------------------------------------------------------


class RateLimiter:
    """Thread-safe global rate limiter: at most ``per_sec`` acquisitions/second.

    Implemented as a moving reservation rather than a sleeping token refill: each
    caller reserves the next free slot under the lock and sleeps outside it, so
    N threads serialize into an evenly spaced request train instead of bursting
    and then idling. Even spacing is what SEC's "carefully monitored" ceiling
    actually asks for -- a burst of 10 inside 100ms averages fine over a second
    and still looks like a spike from the other end.
    """

    def __init__(self, per_sec: float) -> None:
        if per_sec <= 0:
            raise EdgarError(f"rate_limit_per_sec must be positive, got {per_sec!r}")
        self.per_sec = float(per_sec)
        self._interval = 1.0 / self.per_sec
        self._lock = threading.Lock()
        self._next_slot = 0.0

    def acquire(self) -> float:
        """Block until a request may be sent. Returns seconds spent waiting."""
        with self._lock:
            now = time.monotonic()
            slot = max(now, self._next_slot)
            self._next_slot = slot + self._interval
        wait = slot - time.monotonic()
        if wait > 0:
            time.sleep(wait)
            return wait
        return 0.0

    def penalize(self, seconds: float) -> None:
        """Push the next free slot out by ``seconds``, for every waiting thread.

        Called on a 429. Backing off only the thread that got throttled would let
        the other workers keep hammering the endpoint that just asked us to stop,
        which is how a brief limit turns into a ten-minute block.
        """
        with self._lock:
            self._next_slot = max(self._next_slot, time.monotonic() + seconds)


# --------------------------------------------------------------------------
# HTTP client
# --------------------------------------------------------------------------


class EdgarClient:
    """Polite, instrumented HTTP client for sec.gov and data.sec.gov.

    Construct once per stage and share it across that stage's worker threads;
    the limiter inside is what keeps the whole pool under the published ceiling.
    """

    def __init__(self, config: Config, stage: StageRecorder | None = None) -> None:
        import httpx

        user_agent = config.secret("fetch.user_agent_env")
        if not user_agent or not user_agent.strip():
            raise EdgarError(
                "SEC requires a declared User-Agent identifying the requester "
                "(see https://www.sec.gov/os/webmaster-faq). Set "
                f"{config.get('fetch.user_agent_env')} in .env; requests without "
                "it are answered with 403."
            )

        self.stage = stage
        self.max_retries = int(config.get("fetch.max_retries", 5))
        self.backoff_initial = float(config.get("fetch.backoff_initial_sec", 1.0))
        self.backoff_max = float(config.get("fetch.backoff_max_sec", 60.0))
        self.timeout_sec = float(config.get("fetch.timeout_sec", 30))
        self.limiter = RateLimiter(config.get("fetch.rate_limit_per_sec", 8))
        self._throttle_waits = 0.0

        self._client = httpx.Client(
            headers={
                "User-Agent": user_agent.strip(),
                "Accept-Encoding": "gzip, deflate",
            },
            timeout=self.timeout_sec,
            follow_redirects=True,
        )

    def __enter__(self) -> EdgarClient:
        return self

    def __exit__(self, *exc: Any) -> bool:
        self.close()
        return False

    def close(self) -> None:
        self._client.close()

    @property
    def throttle_wait_sec(self) -> float:
        """Total seconds spent waiting on the limiter. Reported to the run log."""
        return round(self._throttle_waits, 3)

    def get(self, url: str) -> httpx.Response:
        """GET with rate limiting and retries. Raises EdgarError when exhausted.

        Every attempt is counted as a request, including retries: a retry is real
        load on SEC, and a run log that counted only successes would understate
        what a flaky run actually cost.
        """
        import httpx

        last_detail = "no attempt made"
        attempts = self.max_retries + 1

        for attempt in range(attempts):
            self._throttle_waits += self.limiter.acquire()
            if self.stage is not None:
                self.stage.request()
            try:
                response = self._client.get(url)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_detail = f"{type(exc).__name__}: {exc}"
            else:
                if self.stage is not None:
                    self.stage.bytes_in(len(response.content))
                if response.status_code == 200:
                    return response
                if response.status_code in (403, 404):
                    # Not transient. 404 means the document is not where the
                    # index said it was; 403 means our declared identity was
                    # rejected. Retrying either just burns rate budget.
                    raise EdgarError(
                        f"HTTP {response.status_code} for {url}"
                        + (
                            " -- SEC rejected the declared User-Agent; check "
                            "EDGAR_USER_AGENT is a real 'Name email' string."
                            if response.status_code == 403
                            else ""
                        )
                    )
                if response.status_code == 429:
                    # Slow the whole pool, not just this thread.
                    delay = self._retry_after(response, attempt)
                    self.limiter.penalize(delay)
                    last_detail = f"HTTP 429 (rate limited), backing off {delay:.1f}s"
                    if attempt + 1 < attempts:
                        time.sleep(delay)
                    continue
                if 500 <= response.status_code < 600:
                    last_detail = f"HTTP {response.status_code}"
                else:
                    raise EdgarError(f"HTTP {response.status_code} for {url}")

            if attempt + 1 < attempts:
                time.sleep(self._backoff(attempt))

        raise EdgarError(
            f"giving up on {url} after {attempts} attempt(s): {last_detail}"
        )

    def get_json(self, url: str) -> Any:
        """GET and parse JSON, with a legible error if the body is not JSON."""
        response = self.get(url)
        try:
            return response.json()
        except ValueError as exc:
            raise EdgarError(
                f"{url} did not return JSON "
                f"(content-type {response.headers.get('content-type')!r}): {exc}"
            ) from exc

    def _backoff(self, attempt: int) -> float:
        """Exponential backoff with jitter, capped at ``fetch.backoff_max_sec``.

        Jitter matters with a shared pool: without it, every worker that failed on
        the same blip retries in the same millisecond and reproduces the blip.
        """
        base = min(self.backoff_initial * (2**attempt), self.backoff_max)
        return base * (0.5 + random.random() / 2.0)

    def _retry_after(self, response: httpx.Response, attempt: int) -> float:
        """Honor Retry-After when SEC sends one, else back off exponentially."""
        raw = response.headers.get("Retry-After")
        if raw:
            try:
                return min(float(raw), self.backoff_max)
            except ValueError:
                pass
        return self._backoff(attempt)


# --------------------------------------------------------------------------
# Identifier and URL construction -- one definition, used by every stage
# --------------------------------------------------------------------------


def cik10(cik: str | int) -> str:
    """Zero-padded 10-digit CIK string, the form data.sec.gov paths expect."""
    digits = str(cik).strip().upper().removeprefix("CIK").lstrip("-").strip()
    if not digits.isdigit():
        raise EdgarError(f"not a CIK: {cik!r}")
    return digits.zfill(10)


def accession_nodash(accession: str) -> str:
    """``0000320193-24-000081`` -> ``000032019324000081`` (Archives path form)."""
    return str(accession).replace("-", "").strip()


def accession_dashed(accession: str) -> str:
    """``000032019324000081`` -> ``0000320193-24-000081`` (canonical form).

    The manifest stores the dashed form because that is how SEC prints it in the
    submissions API and in filing indexes; the undashed form appears only inside
    Archives paths and as a filename stem.
    """
    raw = accession_nodash(accession)
    if len(raw) != 18 or not raw.isdigit():
        raise EdgarError(f"not an 18-digit accession number: {accession!r}")
    return f"{raw[:10]}-{raw[10:12]}-{raw[12:]}"


def primary_doc_url(archives_url: str, cik: str, accession: str, document: str) -> str:
    """URL of any document inside a filing, in the EDGAR Archives.

    Note the CIK is unpadded here while data.sec.gov pads it -- an inconsistency
    in EDGAR itself, centralized in this one function so no stage has to remember
    which convention applies where.
    """
    return (
        f"{archives_url.rstrip('/')}/edgar/data/{int(cik10(cik))}/"
        f"{accession_nodash(accession)}/{document.lstrip('/')}"
    )


def filing_index_url(archives_url: str, cik: str, accession: str) -> str:
    """URL of a filing's index-headers document.

    This is the only place EDGAR publishes an authoritative document TYPE per
    file in a filing. The sibling ``index.json`` looks like the obvious choice
    and is not: its ``type`` field holds an icon filename ("text.gif"), so the
    only thing in it suggesting a file is EX-99.1 is the filename itself, and
    naming a document type from a filename is a guess with no source.
    """
    dashed = accession_dashed(accession)
    return (
        f"{archives_url.rstrip('/')}/edgar/data/{int(cik10(cik))}/"
        f"{accession_nodash(accession)}/{dashed}-index-headers.html"
    )


#: One <DOCUMENT> block of the SGML header, after HTML unescaping.
_DOC_BLOCK_RE = re.compile(r"<DOCUMENT>(.*?)(?=<DOCUMENT>|</SEC-HEADER>|\Z)", re.S | re.I)


def _sgml_field(block: str, name: str) -> str | None:
    """Read one ``<NAME>value`` line from an SGML block.

    SGML header fields are unclosed: the value runs to end of line.
    """
    match = re.search(rf"<{name}>([^\r\n<]*)", block, re.I)
    if match is None:
        return None
    value = match.group(1).strip()
    return value or None


def parse_index_headers(payload: str) -> list[dict[str, Any]]:
    """Parse a filing's index-headers document into per-document records.

    Returns dicts with ``type``, ``sequence``, ``filename`` and ``description``.
    The body embeds the SGML header HTML-escaped inside a <PRE> block, so it is
    unescaped first::

        &lt;DOCUMENT&gt;
        &lt;TYPE&gt;EX-99.1
        &lt;SEQUENCE&gt;2
        &lt;FILENAME&gt;ex991-pressreleasebod.htm

    A block with no FILENAME is returned with ``filename`` None for the caller to
    log and skip. It is never reconstructed -- a fabricated filename would 404 at
    fetch time while looking like real data in the manifest.
    """
    text = html.unescape(payload)
    records: list[dict[str, Any]] = []
    for block in _DOC_BLOCK_RE.findall(text):
        doc_type = _sgml_field(block, "TYPE")
        filename = _sgml_field(block, "FILENAME")
        if doc_type is None and filename is None:
            continue
        records.append(
            {
                "type": doc_type,
                "sequence": _sgml_field(block, "SEQUENCE"),
                "filename": filename,
                "description": _sgml_field(block, "DESCRIPTION"),
            }
        )
    return records


def matches_exhibit_type(doc_type: str | None, prefixes: Iterable[str]) -> bool:
    """Case-insensitive prefix match of an SGML document TYPE.

    Prefix rather than equality so that configuring ``EX-99`` admits EX-99,
    EX-99.1 and EX-99.2 without enumerating them. It still excludes the
    ``EX-101.*`` XBRL taxonomy documents that sit in the same index.
    """
    if not doc_type:
        return False
    upper = doc_type.strip().upper()
    return any(upper.startswith(str(p).strip().upper()) for p in prefixes if str(p).strip())


def as_stored_path(path: str | Path) -> str:
    """Render a path for storage in a ledger: repo-relative, POSIX separators.

    Ledgers are read on whichever host runs the stage, and this harness runs on
    Windows and in WSL against the same tree. A Windows-written
    ``temp\\filings\\x.html`` is, on Linux, a single filename containing
    backslashes -- so every document would look unfetched and the stage would
    re-download the entire corpus. Storing one canonical form removes that.
    """
    path = Path(path)
    try:
        path = path.resolve().relative_to(Path.cwd().resolve())
    except ValueError:
        # Outside the working directory: keep it as given rather than inventing
        # a relative path that would resolve somewhere else.
        pass
    return path.as_posix()


def from_stored_path(value: str) -> Path:
    """Read a path out of a ledger, accepting either separator.

    Accepts the Windows-written form so an existing ledger keeps working without
    a rewrite; ``as_stored_path`` makes everything written from now on portable.
    """
    return Path(str(value).replace("\\", "/"))


_UNSAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def stored_filename(
    accession: str, doc_type: str, document: str, extension: str
) -> str:
    """Name a fetched document on disk.

    Primary documents keep the name they have always had, ``<accession>.html``.
    That is not cosmetic: it is what lets every document fetched before exhibits
    existed still satisfy the skip check, so extending the corpus costs zero
    re-fetching.

    Exhibits are suffixed with their EDGAR filename stem, sanitized. The stem is
    kept rather than the exhibit label so that two exhibits of the same label in
    one filing cannot collide, and so the on-disk name still points back at the
    source document.
    """
    stem = accession_nodash(accession)
    if doc_type == DOC_TYPE_PRIMARY:
        return f"{stem}{extension}"
    # Both separators are normalized explicitly rather than relying on Path,
    # which treats "\" as a separator only on Windows: this harness runs on
    # Windows and in WSL, and a document name is fetched data, so the guard
    # against it escaping out_dir must not depend on the host.
    raw = str(document).replace("\\", "/").rsplit("/", 1)[-1]
    doc_stem = Path(raw).stem or "exhibit"
    safe = _UNSAFE_NAME_RE.sub("_", doc_stem).strip("._-") or "exhibit"
    return f"{stem}-{safe}{extension}"


# --------------------------------------------------------------------------
# Cross-stage CSV schemas
#
# These headers are the contract between stages. They are declared here, once,
# so a stage cannot half-change a schema its neighbour still reads.
# --------------------------------------------------------------------------

#: inputs/universe.csv -- ticker -> CIK, joined from SEC's company_tickers.json.
UNIVERSE_FIELDS: tuple[str, ...] = ("ticker", "cik", "company_name")

#: inputs/manifest.csv -- stage 1 output, stage 2 input.
#:
#: One row per DOCUMENT, not per filing. ``accession`` remains the filing
#: identifier and is therefore no longer unique: a filing contributes its primary
#: document plus one row per qualifying exhibit. The identity of a row is
#: ``(accession, document)``.
MANIFEST_FIELDS: tuple[str, ...] = (
    "cik",
    "ticker",
    "accession",
    "form",
    "filing_date",
    "period_of_report",
    "doc_type",        # primary | exhibit
    "exhibit_label",   # e.g. EX-99.1; blank on primary rows
    "document",        # EDGAR filename, e.g. ex991-pressrelease.htm
    "doc_url",
)

#: Legacy manifest header, before exhibits became rows. Recognized only to give a
#: precise error telling the reader to re-run discover.
LEGACY_MANIFEST_FIELDS: tuple[str, ...] = (
    "cik",
    "ticker",
    "accession",
    "form",
    "filing_date",
    "period_of_report",
    "primary_doc_url",
)

#: temp/filings/fetched.csv -- stage 2 ledger. Makes skip-before-request work
#: across runs and survives a crash mid-stage. Keyed by ``(accession, document)``.
LEDGER_FIELDS: tuple[str, ...] = (
    "accession",
    "document",
    "doc_type",
    "path",
    "status",
    "bytes",
    "content_type",
    "sha256",
    "fetched_at",
)

#: Ledger header written before exhibits existed. Every row in a file with this
#: header is a primary document, because that is all stage 2 ever fetched --
#: which is what makes the migration in tools/fetch.py sound rather than a guess.
LEGACY_LEDGER_FIELDS: tuple[str, ...] = (
    "accession",
    "path",
    "status",
    "bytes",
    "content_type",
    "sha256",
    "fetched_at",
)

DOC_TYPE_PRIMARY = "primary"
DOC_TYPE_EXHIBIT = "exhibit"


def _check_header(path: Path, got: Iterable[str] | None, want: tuple[str, ...]) -> None:
    got_tuple = tuple(got or ())
    if got_tuple != want:
        raise EdgarError(
            f"{path} has header {got_tuple!r}, expected {want!r}. "
            f"Re-run the stage that produces it rather than editing it by hand."
        )


def load_universe(path: str | Path) -> list[dict[str, str]]:
    """Read inputs/universe.csv, validating the header and normalizing CIKs."""
    path = Path(path)
    if not path.is_file():
        raise EdgarError(
            f"{path} not found. Build it first: python -m tools.build_universe"
        )
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        _check_header(path, reader.fieldnames, UNIVERSE_FIELDS)
        rows = [dict(row) for row in reader]
    if not rows:
        raise EdgarError(f"{path} has a header but no rows.")
    for row in rows:
        row["cik"] = cik10(row["cik"])
        row["ticker"] = row["ticker"].strip().upper()
    return rows


def write_universe(path: str | Path, rows: list[dict[str, Any]]) -> int:
    """Write inputs/universe.csv. Returns bytes written."""
    return _write_csv(path, UNIVERSE_FIELDS, rows)


def read_manifest(
    path: str | Path, *, allow_legacy: bool = False
) -> list[dict[str, str]]:
    """Read inputs/manifest.csv, validating the header.

    Stage 2 reads this rather than re-querying submissions -- Rule 1: any stage
    must be re-runnable without repeating the previous one.

    ``allow_legacy`` upgrades a pre-exhibit manifest in memory instead of
    refusing it. Only the audit sets it, because the audit's job includes
    describing a corpus fetched *before* exhibits were part of it; a stage that
    writes or fetches must have the real schema.
    """
    path = Path(path)
    if not path.is_file():
        raise EdgarError(
            f"{path} not found. Build it first: python -m tools.discover"
        )
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        got = tuple(reader.fieldnames or ())
        rows = [dict(row) for row in reader]

    if got == MANIFEST_FIELDS:
        return rows
    if got == LEGACY_MANIFEST_FIELDS:
        if not allow_legacy:
            raise EdgarError(
                f"{path} uses the pre-exhibit manifest schema (one row per filing). "
                f"The manifest now carries one row per document. Re-run "
                f"`python -m tools.discover` to rebuild it; already-fetched "
                f"documents are unaffected and will be skipped."
            )
        return [_upgrade_legacy_manifest_row(r) for r in rows]
    _check_header(path, got, MANIFEST_FIELDS)
    return rows


def _upgrade_legacy_manifest_row(row: dict[str, str]) -> dict[str, str]:
    """Map a pre-exhibit manifest row onto the per-document schema.

    Every legacy row is a primary document -- that is all the manifest could
    hold. ``document`` is the last path segment of ``primary_doc_url``, which is
    not a guess: that URL was built from the submissions API's ``primaryDocument``
    field, so its basename is the filename EDGAR published.
    """
    url = (row.get("primary_doc_url") or "").strip()
    return {
        "cik": row.get("cik", ""),
        "ticker": row.get("ticker", ""),
        "accession": row.get("accession", ""),
        "form": row.get("form", ""),
        "filing_date": row.get("filing_date", ""),
        "period_of_report": row.get("period_of_report", ""),
        "doc_type": DOC_TYPE_PRIMARY,
        "exhibit_label": "",
        "document": url.rsplit("/", 1)[-1] if url else "",
        "doc_url": url,
    }


def manifest_key(row: dict[str, Any]) -> tuple[str, str]:
    """Identity of a manifest row: the filing, plus which document within it."""
    return (str(row.get("accession") or ""), str(row.get("document") or ""))


def write_manifest(path: str | Path, rows: list[dict[str, Any]]) -> int:
    """Write inputs/manifest.csv in deterministic order. Returns bytes written.

    Sorted by ticker, filing_date descending, accession, then primary-before-
    exhibit, then document. Re-running discover over an unchanged window
    therefore produces a byte-identical file, which makes "did the corpus
    change?" answerable by diff rather than by inspection.

    The last two sort keys are what keep that guarantee now that a filing
    contributes several rows: ordering by accession alone would leave the rows
    within a filing at the mercy of dict iteration order.
    """
    ordered = sorted(
        rows,
        key=lambda r: (
            str(r.get("ticker") or ""),
            _descending(str(r.get("filing_date") or "")),
            str(r.get("accession") or ""),
            0 if r.get("doc_type") == DOC_TYPE_PRIMARY else 1,
            str(r.get("document") or ""),
        ),
    )
    return _write_csv(path, MANIFEST_FIELDS, ordered)


def _descending(date_str: str) -> str:
    """Sort key inverting an ISO date, so dates sort newest-first ascending."""
    return "".join(chr(ord("9") - int(ch)) if ch.isdigit() else ch for ch in date_str)


def _write_csv(
    path: str | Path, fields: tuple[str, ...], rows: Iterable[dict[str, Any]]
) -> int:
    """Write a CSV atomically with LF endings. Returns bytes written.

    Atomic because a crash partway through must not leave a truncated file that
    a later stage would happily read as complete. LF explicitly, so the file is
    byte-identical whether it was produced on Windows or in WSL -- the harness
    runs in both and a line-ending diff would be pure noise.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _blank_if_none(row.get(k)) for k in fields})
    size = tmp.stat().st_size
    tmp.replace(path)
    return size


def _blank_if_none(value: Any) -> Any:
    """Render a missing value as an empty cell.

    Rule 4: a missing field stays missing. It is written blank and logged by the
    caller, never back-filled with a placeholder that would later read as data.
    """
    return "" if value is None else value


def iter_recent_filings(submissions: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Yield one dict per filing from a submissions ``filings.recent`` block.

    The API returns columnar data -- parallel arrays keyed by field name, not a
    list of records -- so rows are reconstructed by zipping. Arrays are trusted
    to be equal length only as far as the shortest one: a short array would
    otherwise silently pair a form with the wrong date, which is exactly the kind
    of corruption that is invisible until retrieval results look strange.
    """
    yield from _iter_columnar(submissions.get("filings", {}).get("recent", {}))


def iter_shard_filings(shard: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Yield one dict per filing from an older ``filings.files[]`` shard.

    Shards are the same columnar shape as ``recent`` but are the whole document
    rather than a nested block.
    """
    yield from _iter_columnar(shard)


def _iter_columnar(block: dict[str, Any]) -> Iterator[dict[str, Any]]:
    if not isinstance(block, dict) or not block:
        return
    columns = {k: v for k, v in block.items() if isinstance(v, list)}
    if not columns:
        return
    length = min(len(v) for v in columns.values())
    for i in range(length):
        yield {k: v[i] for k, v in columns.items()}
