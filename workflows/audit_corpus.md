# Workflow — Corpus audit

## Objective

Answer one question about what is on disk: is it structurally capable of
supporting retrieval, or are there documents that are pointers to content we do
not have?

This is not a pipeline stage. It is a check run *between* stages — in particular
before stage 3, because chunking and indexing a corpus of pointers would produce
an index that looks complete and retrieves nothing useful.

## Inputs

| Input | Source |
|---|---|
| `inputs/manifest.csv` | `workflows/01_discover.md` |
| `temp/filings/fetched.csv` + the documents | `workflows/02_fetch.md` |
| `audit.*` | `inputs/config.yaml` — thresholds, patterns, sample size and seed |

## Tool

```
python -m tools.audit_corpus [--json-only] [--set audit.short_item_threshold=1000]
```

**Reads local files only — no network, so it runs without approval.** Its run log
correctly shows `requests: 0`; that zero is what distinguishes it from a stage.

It reads both the current and the pre-exhibit manifest/ledger schemas on purpose:
describing a corpus fetched *before* exhibits were part of it is one of its jobs.

## Outputs

| Output | Contents |
|---|---|
| stdout | Human-readable report: per form/doc-type, per ticker, exhibit resolution, and a sample |
| `outputs/audit/<run_id>.json` | The same report plus a per-document record, so two audits can be diffed |
| `outputs/runs/<run_id>.jsonl` | `requests: 0`, items = documents scored |

## What it measures, and why

**Body text volume alone answers nothing.** An 8-K whose entire narrative is
"see the press release attached as Exhibit 99.1" still carries 2,000–3,000
characters of mandatory cover page — registrant, address, checkbox paragraphs,
signature. A naive character threshold therefore passes every document and
detects nothing. On this corpus the minimum was 2,687 characters and *zero of
184* fell under a 2,000-character threshold, while 114 were pointers.

**So the audit splits the text.** It cuts at the first item anchor and again at
the signature block; the span between is the filing's own narrative, as distinct
from the cover page around it and from content incorporated by reference. A short
item narrative next to an exhibit reference is the shape of a pointer.

**Exhibit resolution is the direct measure.** For every EX-99.x a document
references, is that exhibit present on disk? This is the number that says whether
the gap closed.

The item split applies only where the configured `audit.item_pattern` matched.
That pattern is the 8-K `N.NN` form and finds nothing in a 10-Q, so 10-Q item
statistics read `n/a` rather than `0` — reporting a zero there would say "these
filings have no narrative" when the truth is "this measure does not apply".

## Procedure

1. Run `python -m tools.audit_corpus`.
2. Read **exhibit resolution** first: `UNRESOLVED` is the count of documents that
   reference an exhibit not on disk. Zero is the goal.
3. Check `documents_missing_on_disk` is 0 — a manifest row with no fetched
   document means stage 2 has work left.
4. Compare against the previous `outputs/audit/*.json` to see what changed.

## Baseline — before exhibits were fetched

Recorded so the next run has something to be compared against:

| Measure | Value |
|---|---|
| documents scored | 224 |
| 8-K primary documents | 184 |
| referencing an Exhibit 99.x | 114 |
| ...resolved | **0** |
| Item 2.02 documents | 60 |
| ...with an unresolved reference | **60** |
| 8-K body chars (min / median / max) | 2,687 / 4,848 / 28,508 |
| 10-Q body chars (min / median / max) | 84,537 / 149,999 / 441,701 |

## Edge cases

| Case | Handling |
|---|---|
| Manifest predates the exhibit schema | Read and upgraded in memory; the audit still runs. Stages that write or fetch refuse it instead. |
| Ledger predates the exhibit schema | A primary row matches an entry filed under an empty document name. An exhibit row never falls back that way, because an empty-named entry can only be a primary. |
| Manifest row with no fetched document | Counted as `documents_missing_on_disk` and logged as an error. |
| Malformed HTML | Text recovered so far is used; the document is not discarded over markup we do not control. |

## Rule notes

- **Rule 6**: this tool reports structure and counts. It does not characterize
  what any filing says, and it emits no conclusion or recommendation.
  "This document is 3,604 characters and references an exhibit that is not on
  disk" is a fact about files.
- **Rule 3**: thresholds, patterns, sample size and seed are all config. The seed
  is fixed so the reported sample is reproducible across runs and machines.
