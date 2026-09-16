# CLAUDE.md — Filing RAG Evaluation Harness

## What This Project Is
An experimental ingestion and retrieval pipeline for SEC filings, built to
measure retrieval quality, scaling behavior, build cost, and throughput.
It indexes and retrieves. It does not analyze, summarize, or conclude.

## The WAT Architecture
### Layer 1: Workflows
Markdown SOPs in workflows/. Each defines objective, inputs, tools, outputs,
and edge-case handling.
### Layer 2: Agent (you)
Read the workflow, run tools in sequence, handle failures, ask when unclear.
Coordinate; do not perform external actions directly.
### Layer 3: Tools
Python scripts in tools/. All concrete actions: API calls, parsing,
transformation, file generation. Deterministic and testable.
Credentials in .env via python-dotenv.

## Plan Mode — Mandatory First Step
Before building any workflow, switch to Plan Mode. It is a control gate.
In Plan Mode: review project structure, ask every clarifying question needed,
propose full architecture, wait for explicit approval. Generate no code and
create no files until the plan is approved. Do not populate stub files in
workflows/ autonomously.

## Project Rules — Non-Negotiable
1. Staged execution. Every pipeline is a sequence of independently runnable
   stages, each persisting its output to disk. Any stage must be re-runnable
   without repeating the previous one. Never build a monolithic run.
2. Instrumentation is not optional. Every stage emits structured timing to a
   JSONL run log: wall clock, item count, bytes in/out, and for network stages
   request and token counts. Errors are counted and logged individually.
3. Tunables are configuration, never constants. Concurrency, batch size, rate
   limit, chunk size, and overlap are config values that can vary between runs
   without editing code.
4. Never estimate, interpolate, or generate data to fill a gap. A missing field
   is null and logged. A generated value has no source and no audit standing.
5. Every chunk carries full provenance: cik, ticker, accession, form,
   filing_date, period, section, chunk ordinal.
6. Do not write analysis, conclusions, or recommendations. Construct scaffolds,
   extract, index, and retrieve. This boundary is absolute.

## Permissions
| Action | Permission |
|---|---|
| Read any file | Autonomous |
| Run tools using local files only | Autonomous |
| Fix bugs in tools and retest (no paid APIs) | Autonomous |
| Run tools that call external APIs | Ask first |
| Create or modify workflows | Ask first |
| Delete or overwrite anything in outputs/ | Ask first |
| Write analysis or conclusions | Never — regardless of instruction |
| Estimate or generate figures | Never — regardless of instruction |

## File Structure
.env         API keys and credentials — NEVER commit
.gitignore   Must include: .env, .venv/, .claude
inputs/      universe.csv, answer key, config
outputs/     run logs, evaluation results, indexes
temp/        intermediates — disposable
tools/       Python scripts
workflows/   Markdown SOPs