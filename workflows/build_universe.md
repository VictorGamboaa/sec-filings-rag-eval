# Workflow — Build the ticker universe

## Objective

Produce `inputs/universe.csv` mapping each ticker in scope to its SEC CIK, with
every CIK sourced from SEC's own published mapping rather than from recall.

This is not a pipeline stage. It is the prerequisite that makes stage 1
addressable: `discover` queries by CIK, and a wrong CIK does not fail — it builds
a complete, plausible corpus of the wrong company's filings.

## Inputs

| Input | Source |
|---|---|
| `universe.tickers` | `inputs/config.yaml` — the tickers in scope |
| `universe.tickers_url` | `inputs/config.yaml` — SEC's `company_tickers.json` |
| `EDGAR_USER_AGENT` | `.env`, via `fetch.user_agent_env` |

## Tool

```
python -m tools.build_universe [--dry-run] [--from-cache] [--set k=v ...]
```

Calls an external API. **Ask before running** (per the permissions table in
CLAUDE.md). `--dry-run` and `--from-cache` make no request and need no approval.

## Outputs

| Output | Contents |
|---|---|
| `inputs/universe.csv` | `ticker,cik,company_name` — one row per ticker in scope |
| `temp/company_tickers.json` | The bytes SEC served, kept as the audit trail for the join |
| `outputs/runs/<run_id>.jsonl` | `run_start` / `stage_start` / `stage_end` / `run_end`, plus one `error` line per failure |

`cik` is zero-padded to 10 digits, the form `data.sec.gov` paths expect.

## Procedure

1. Confirm `EDGAR_USER_AGENT` resolves: `python -m tools.config --check-env`.
2. `python -m tools.build_universe --dry-run` — confirm the ticker list and paths.
3. Ask for approval, then run `python -m tools.build_universe`.
4. Read back the printed table and confirm each company name matches the ticker
   you expected. This is the only human check in the chain that would catch a
   ticker that legitimately resolves to a different company than intended.
5. Confirm the run log's `stage_end` shows `items` equal to the ticker count and
   `errors: 0`.

## Edge cases

| Case | Handling |
|---|---|
| Ticker absent from SEC's file | **Hard failure**, naming the ticker. `universe.csv` is not written. No nearest match is substituted, and the ticker is not silently dropped — a short universe that looked complete would misstate the corpus. |
| Ticker matches more than one record | **Hard failure**, naming the ticker and every candidate CIK. Ambiguity cannot be resolved without guessing. |
| `company_tickers.json` shape changed | **Hard failure** naming what was found. The join never proceeds on an unvalidated structure. |
| Fetch succeeded, parse failed | The raw bytes are cached *before* parsing, so the response that broke the parse is on disk for inspection. |
| 403 from SEC | The declared User-Agent was rejected. Check `EDGAR_USER_AGENT` is a real `Name email@domain` string. Not retried — retrying burns rate budget against a request that cannot succeed. |
| Re-run needed without network | `--from-cache` re-joins against the cached file. |

## Rule notes

- **Rule 4** is the whole point of this workflow: a CIK with no fetched source is
  a generated value and has no audit standing.
- **Rule 3**: the ticker list is config. Changing the universe is
  `--set universe.tickers='["AAPL","MSFT"]'` or a YAML edit, never a code change.
