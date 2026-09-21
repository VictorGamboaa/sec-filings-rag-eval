# Workflow — Stage 1: discover

## Objective

From the ticker universe, produce `inputs/manifest.csv`: one row per 10-Q and
8-K whose **filing date** falls inside the configured window.

This stage decides what the corpus *is*. Every later stage operates on exactly
the rows it emits.

## Scope is a filing-date window, never a fiscal label

The five names in this universe do not share a fiscal calendar — Albertsons runs
an offset fiscal year. Selecting by a label such as "Q2 FY25" would therefore
pull different real periods for different companies, and any cross-name
retrieval comparison built on that corpus would be measuring the calendar, not
the retriever.

Selection is purely `discover.date_from <= filing_date <= discover.date_to`.
`period_of_report` is carried into the manifest as **metadata only** and is never
used to include or exclude a filing. Expect `period_of_report` to differ across
names for filings made in the same weeks — that divergence is correct, and seeing
it in the output is the check that the window did the selecting.

Narrowing the window is also how the scaling curve gets produced, which is the
second reason it is config rather than anything fixed.

## Inputs

| Input | Source |
|---|---|
| `inputs/universe.csv` | `workflows/build_universe.md` |
| `discover.date_from` / `date_to` | `inputs/config.yaml` — `date_to: null` means today |
| `discover.forms` | `inputs/config.yaml` — matched **exactly** |
| `discover.concurrency`, `fetch.rate_limit_per_sec` | `inputs/config.yaml` |
| `EDGAR_USER_AGENT` | `.env` |

## Tool

```
python -m tools.discover [--dry-run] [--limit N] [--set k=v ...]
```

Calls an external API. **Ask before running.** `--dry-run` makes no request.

## Outputs

| Output | Contents |
|---|---|
| `inputs/manifest.csv` | `cik,ticker,accession,form,filing_date,period_of_report,doc_type,exhibit_label,document,doc_url` |
| `outputs/runs/<run_id>.jsonl` | wall clock, item count, bytes in/out, request count, error count, plus one `error` line per failure |

Written atomically and sorted (ticker asc, filing_date desc, accession,
primary-before-exhibit, document), so re-running over an unchanged window
produces a **byte-identical** file. "Did the corpus change?" is answerable by
`diff`, not by inspection.

## One row per DOCUMENT, not per filing

`accession` identifies the **filing** and is no longer unique: a filing
contributes its primary document plus one row per qualifying exhibit. The
identity of a row is `(accession, document)`.

| Column | Meaning |
|---|---|
| `doc_type` | `primary` or `exhibit` |
| `exhibit_label` | `EX-99.1` etc.; blank on primary rows |
| `document` | the filename EDGAR published |
| `doc_url` | full Archives URL (was `primary_doc_url`) |

### Where exhibit filenames come from

For each filing whose form is in `discover.exhibit_forms`, the stage fetches the
filing's **index-headers** document and reads the SGML `<TYPE>` / `<FILENAME>`
pairs.

Two wrong sources, both checked and rejected:

- **`index.json`** looks like the obvious choice and is not. Its `type` field
  holds an *icon filename* (`text.gif`), so the only thing in it suggesting a
  file is EX-99.1 is the filename — and naming a document type from a filename
  is a guess with no source.
- **Hyperlinks in the primary document.** Only 16 of 184 8-Ks in this corpus
  link their exhibits; 98 name "Exhibit 99" in prose with no link at all.

`EX-101.*` XBRL taxonomy documents appear in the same index, which is why
`discover.exhibit_types` is a prefix match on `EX-99` rather than a wildcard.

## Procedure

1. `python -m tools.discover --dry-run` — confirm the resolved window (note the
   effective `date_to` when it is configured `null`), the form list, and the
   planned requests.
2. Ask for approval, then run `python -m tools.discover`.
3. Check the printed per-ticker counts. A ticker with zero rows is reported as an
   error line, not passed over quietly.
4. Verify in the run log's `stage_end`:
   - `date_to_effective` — the actual upper bound used
   - `filings_by_form` — only the configured forms appear
   - `window_truncated: false` and `failed_ciks: []`
   - `missing_primary_doc` — how many rows have a null URL
5. **Check the exhibit tripwire** (below).
6. Re-run and confirm the manifest is byte-identical.

## The exhibit tripwire

Every 8-K gets an index request, because whether it *has* a qualifying exhibit is
exactly what the index answers. Only some produce an extra document row. The run
log records the split:

| Field | Expected on this corpus |
|---|---|
| `exhibit_index_fetched` | 184 |
| `filings_with_qualifying_exhibit` | ~114 |
| `filings_without_qualifying_exhibit` | **~70** |
| `exhibit_index_failed` | 0 |
| `exhibit_rows` | ≥ 114 |
| `exhibit_types_seen` | every `<TYPE>` encountered, with counts |

**If `filings_without_qualifying_exhibit` comes back much above ~70, the
exhibit-type matching is wrong — not the filings being exhibit-free.** Read
`exhibit_types_seen` to find out which types exist and what the prefix failed to
match, then adjust `discover.exhibit_types`.

The ~114 is an *independent* cross-check, not an identity: it comes from
`tools.audit_corpus` running a text regex over primary documents, while these
counts come from EDGAR's own document types. Small divergence is expected — a
filing may discuss an exhibit filed elsewhere, or carry an EX-99 the regex
missed. A large divergence means the matching is broken.

`exhibit_index_failed` is counted **separately** on purpose: a failed request is
not evidence that a filing has no exhibit, and conflating the two would corrupt
the tripwire. A non-zero value exits non-zero, for the same reason a failed CIK
does — the manifest is short by an unknown number of exhibits.

## Edge cases

| Case | Handling |
|---|---|
| Window predates `filings.recent` | `filings.recent` covers roughly the last year or 1000 filings. The stage detects that the window starts before the oldest record there and, with `discover.follow_older_shards: true`, fetches the older `files[]` shards that overlap the window and merges them. **Relevant at the default 2024-01-01 window for an active 8-K filer.** |
| `follow_older_shards: false` and shards exist | The manifest is short. This is logged as an error and flagged `window_truncated: true` — never silent, because a short manifest reads as "this filer files less" rather than "we did not ask for it all". |
| A shard request fails | Logged with the shard name, that CIK marked truncated, the rest of the run continues. |
| Shard's advertised date range unreadable | Treated as overlapping and fetched. An extra request costs far less than a quietly incomplete corpus. |
| `primaryDocument` empty | `primary_doc_url` is written **blank** and one error line is logged. The filename is not reconstructed — a guessed URL would 404 at fetch time while looking like real data in the manifest (Rule 4). |
| `reportDate` empty | `period_of_report` is blank and logged. Never inferred from the filing date. |
| `filingDate` empty | The filing cannot be placed in the window, so it is excluded and logged. |
| Amendments (`10-Q/A`) | Excluded unless `discover.forms` names them. An amendment is a different document; mixing it in silently would confound any measurement over the corpus. |
| One CIK fails after retries | Logged and skipped; the other CIKs still produce rows. The stage **exits non-zero**, so a partial manifest is never mistaken for a complete one. |
| Same filing in both `recent` and a shard | De-duplicated on `(accession, document)`; the count is reported as `duplicates_dropped`. Primaries are de-duplicated *before* indexing, so a duplicate never costs a second index request. |
| A filing's index request fails | Logged with the accession, counted as `exhibit_index_failed`, and the stage exits non-zero. The filing keeps its primary row; only its exhibits are missing. |
| An exhibit `<DOCUMENT>` block has no `<FILENAME>` | Logged and skipped. The filename is never reconstructed — a guessed one would 404 at fetch time while looking like real data (Rule 4). |
| Widening or narrowing exhibit scope | `--set discover.exhibit_types='["EX-99","EX-10"]'` or `--set discover.exhibit_forms='["8-K","10-Q"]'`. Rule 3: a corpus-definition change is config, not code. |

## Rule notes

- **Rule 1**: output is persisted, so stage 2 runs off this file and never
  re-queries submissions.
- **Rule 2**: `stage_end` carries wall clock, items, bytes in/out, requests and
  errors; each failure also gets its own `error` line.
- **Rule 5**: every field stage 3 needs for chunk provenance originates here.
