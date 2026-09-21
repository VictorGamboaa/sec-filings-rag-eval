# Workflow — Stage 2: fetch

## Objective

Download each manifest filing's primary document into `temp/filings/`, named by
accession, under bounded concurrency and a configurable request rate.

## Inputs

| Input | Source |
|---|---|
| `inputs/manifest.csv` | `workflows/01_discover.md` |
| `fetch.concurrency` | `inputs/config.yaml` — how many requests in flight |
| `fetch.rate_limit_per_sec` | `inputs/config.yaml` — requests per second, process-wide |
| `fetch.max_retries`, `backoff_initial_sec`, `backoff_max_sec`, `timeout_sec` | `inputs/config.yaml` |
| `EDGAR_USER_AGENT` | `.env` |

## Tool

```
python -m tools.fetch [--dry-run] [--limit N] [--force] [--set k=v ...]
```

Calls an external API. **Ask before running.** `--dry-run` makes no request.

## Outputs

| Output | Contents |
|---|---|
| `temp/filings/<accession_nodash>.html` (or `.txt`) | A **primary** document as served |
| `temp/filings/<accession_nodash>-<document stem>.html` | An **exhibit** as served |
| `temp/filings/fetched.csv` | Ledger: `accession,document,doc_type,path,status,bytes,content_type,sha256,fetched_at` |
| `outputs/runs/<run_id>.jsonl` | wall clock, items, bytes in/out, requests, errors, plus one `error` line per failure |

## One row per document

The manifest carries one row per document, so this stage fetches primaries and
exhibits alike. Two consequences:

- The skip key is `(accession, document)`. A filing's exhibit is not considered
  fetched because its primary document is present.
- **Primary documents keep the name they have always had**,
  `<accession_nodash>.html`. That is not cosmetic: it is what lets every document
  fetched before exhibits existed still satisfy the skip check, so extending the
  corpus costs zero re-fetching.

Exhibits are named with their EDGAR filename stem rather than their label, so two
exhibits sharing a label in one filing cannot collide, and the on-disk name still
points back at the source document.

## Ledger migration (one time)

A ledger written before exhibits existed has a 7-column header and no `document`.
Every row in such a file is a primary document — not by assumption, but because
fetching `primary_doc_url` was the only thing this stage could do.

On load, each legacy row is matched to the manifest row for its accession with
`doc_type == primary` and takes that row's `document` value. The filename comes
from the manifest, so the value recorded is the one EDGAR published rather than
one derived from our own on-disk name. The ledger is then rewritten in the new
format, and `ledger_migrated_rows` records how many moved.

A legacy row whose accession has no primary manifest row keeps a blank `document`
and is logged individually — it is never guessed.

## The two knobs are independent

`fetch.concurrency` sets how many requests are in flight. `fetch.rate_limit_per_sec`
sets how fast they may be issued, enforced by a **single process-wide limiter
shared by every worker**. Raising concurrency therefore cannot raise the achieved
request rate — which is both what SEC requires and what makes a throughput sweep
interpretable, since only one variable moves at a time.

SEC publishes a ceiling of 10 requests/second, and states it applies "regardless
of the number of machines used to submit requests"
([Internet Security Policy](https://www.sec.gov/about/privacy-information#security);
see also the [Webmaster FAQ](https://www.sec.gov/os/webmaster-faq): "our current
maximum access rate is 10 requests per second"). The configured default of 8
leaves headroom. **Do not raise it above 10.**

## Skip-before-request

Every accession already in the ledger as `status=ok`, whose file is still present
at the recorded size, is skipped **before the request is constructed**. A re-run
over an unchanged manifest issues zero HTTP requests.

That is directly verifiable: re-run immediately and confirm the run log shows
`requests: 0` with `skipped: N`. A zero there is the proof the check really
precedes the request, rather than discarding a response after paying for it.

## Procedure

1. `python -m tools.fetch --dry-run` — confirm how many would be fetched vs skipped.
2. Ask for approval, then run `python -m tools.fetch --limit 3` first. Open one of
   the three files and confirm it is the filing it claims to be.
3. Ask for approval, then run the full `python -m tools.fetch`.
4. Re-run immediately; confirm `requests: 0` and `skipped: N` in the run log.
5. Check `stage_end` for `fetched`, `skipped`, `failed`, `no_url`, `bytes_in`,
   `bytes_out` and `throttle_wait_sec`.

## Edge cases

| Case | Handling |
|---|---|
| **429 rate limited** | The **whole pool** backs off, not just the throttled worker — the other workers would otherwise keep hitting the endpoint that just asked us to stop. `Retry-After` is honored when sent, else exponential backoff with jitter up to `backoff_max_sec`. Exceeding SEC's ceiling costs a ~10-minute cool-off before access resumes, and SEC reserves the right to block the IP, so a 429 is treated as a signal to slow down, not a transient blip. |
| 403 | Not retried. The declared User-Agent was rejected; retrying burns rate budget on a request that cannot succeed. |
| 404 | Not retried. The document is not where the submissions index said it was. Logged with the accession and URL. |
| 5xx / timeout | Retried with jittered exponential backoff up to `fetch.max_retries`. Jitter matters: without it every worker that failed on one blip retries in the same millisecond and reproduces it. |
| Manifest row with blank `primary_doc_url` | Nothing to fetch. Recorded as `status=no_url` and logged, so the gap is visible in this stage's own error count rather than requiring a reader to correlate two run logs. |
| Unexpected content type | Logged as an error and **not written**. There is no PDF path: EDGAR serves HTML or text, and parsing PDFs would be pure overhead here. Accepted types are config (`fetch.accept_content_types`). |
| Process killed mid-write | Documents are written to a `.part` file and renamed, so a truncated file never lands at the real path. |
| File truncated or deleted after a successful fetch | The skip check compares the on-disk size against the ledger, so a damaged file is re-fetched rather than skipped forever. |
| Previous run recorded `status=error` | Not skipped. Retried on the next run. |
| Need a clean re-fetch | `--force` ignores the ledger entirely. |
| Any document failed after retries | The stage **exits non-zero**. The ledger records which. |
| Two documents in one filing sanitize to the same on-disk name | The second is refused and logged. Writing both would silently leave one document holding the other's bytes, which no later stage could detect. Names already on disk are claimed too, so a collision with an existing file is caught as well as one within the run. |
| Document name contains path separators | Reduced to its last segment before sanitizing, on both `/` and `\` explicitly rather than via `Path` (whose behaviour differs between Windows and WSL). A fetched document's name must not be able to steer where bytes land. |

## Rule notes

- **Rule 1**: reads the manifest from disk; never re-queries the submissions API.
  Re-runnable in isolation, and idempotent within itself.
- **Rule 2**: retries are counted as requests. A run log that counted only
  successes would understate what a flaky run actually cost SEC and us.
- **Rule 4**: a filing that could not be fetched is absent and logged. No
  placeholder document is ever written.
