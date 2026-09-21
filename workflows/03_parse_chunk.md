# Workflow — Stage 3: parse + chunk

## Objective

Turn each fetched document into chunks that carry full provenance: strip the
markup, split on the configured section anchors, then split any oversized
section on the token budget.

## Inputs

| Input | Source |
|---|---|
| `inputs/manifest.csv` | `workflows/01_discover.md` |
| `temp/filings/` + `fetched.csv` | `workflows/02_fetch.md` |
| `parse.section_anchors`, `min_section_chars`, `preamble_label` | `inputs/config.yaml` |
| `chunk.unit / size / overlap / snap_to_sentence` | `inputs/config.yaml` |

## Tool

```
python -m tools.parse_chunk [--dry-run] [--limit N] [--force] [--set k=v ...]
```

**Local files only — no network, so it runs without approval.** The run log
shows `requests: 0`.

Run it in WSL at `~/dev/rag-test`, like every stage.

## Outputs

| Output | Contents |
|---|---|
| `temp/chunks/<accession>[-<document>].jsonl` | One JSON object per chunk |
| `temp/chunks/parsed.csv` | Ledger: `accession,document,doc_sha256,config_fingerprint,chunks,path,parsed_at` |
| `outputs/runs/<run_id>.jsonl` | wall clock, chunk count, bytes in/out, errors, and the section distribution |

### Chunk record

Every field of `ChunkProvenance` — `cik, ticker, accession, form, filing_date,
period, section, chunk_ordinal` — constructed **through the frozen dataclass**, so
a field that went missing upstream fails at construction rather than reaching the
index. Plus `chunk_id`, `text`, `n_tokens`, `char_span`, and `doc_type`,
`document`, `exhibit_label`.

Those last three are extra fields on the record, **not** additions to
`ChunkProvenance`: the dataclass predates exhibits and has no field naming which
document within a filing a chunk came from, and without one a chunk from an 8-K
primary is indistinguishable from one from its EX-99.1.

`char_span` comes from the tokenizer's own offset mapping, never from re-joining
decoded tokens — decoding does not round-trip (WordPiece loses the `##` join), and
a chunk whose text cannot be located in its source document has no provenance.

## The token budget is not arbitrary

`BAAI/bge-small-en-v1.5` accepts **512 tokens** — verified against the model's
published `sentence_bert_config.json` and `config.json`, not recalled. Past that,
embedding backends truncate **silently**: a well-formed vector comes back for the
first 512 tokens with nothing to indicate the rest was dropped.

`chunk.size` is therefore 450, leaving headroom for `[CLS]`/`[SEP]`. Stage 4
refuses any chunk over the embedder's declared `max_seq_tokens` rather than
letting it be truncated.

Measuring in `chunk.unit: tokens` uses the embedding model's own tokenizer, which
is the only way this budget means anything. `chunk.unit: chars` is available for
exercising the pipeline without a model.

## The section distribution — read this before embedding

The stage prints, and records in the run log, chunk counts by section label per
form and document type. Three buckets, kept distinct because they mean different
things:

| bucket | meaning |
|---|---|
| a named label | an anchor matched |
| `_default` | the form had no anchor list of its own and the fallback matched |
| `(unsectioned)` | **no anchor matched** |

The third is the health signal. Anchors are regexes over text extracted from
markup nobody controls; when they stop matching, **nothing fails**. Chunks are
still produced, the index still builds, retrieval merely gets worse. This report
is the only place that becomes visible.

### Baseline, measured on the corpus this was built against

| group | expected unsectioned char share |
|---|---|
| 10-Q/primary | **≈ 2%** |
| 8-K/primary | **≈ 42%** — the cover page precedes the first Item; not a failure |
| 8-K/exhibit | **≈ 0%** — labelled from `exhibit_label` |

A 10-Q share far above 2% means the anchors need attention. Tune them in
`parse.section_anchors` and re-run; the fingerprint change re-parses everything
automatically.

## Skip behaviour

A document is skipped when **both** its fetch-ledger `sha256` and the current
`config_fingerprint` match, and its output file is still present.

- Re-fetching a document re-parses it (its sha changed).
- Changing any anchor, threshold, chunk size, overlap, snap setting **or the
  tokenizer** re-parses everything (the fingerprint changed).
- Nothing else re-parses.

The tokenizer is in the fingerprint because a token count from one tokenizer is
not comparable with a count from another; chunk sizes would change while every
recorded number stayed the same.

`--force` ignores the ledger entirely.

## Procedure

1. `python -m tools.parse_chunk --dry-run` — confirm the fingerprint, the unit
   and how many documents would be parsed.
2. Run `python -m tools.parse_chunk`.
3. **Read the section distribution against the baseline above.** Do this before
   running stage 4; an index built on bad sectioning costs a full re-embed.
4. Confirm in `stage_end`: `chunks`, `skipped`, `failed`, `not_fetched`,
   `unsectioned_char_share`, and `requests: 0`.
5. Re-run and confirm `skipped` equals the document count and `parsed: 0`.

## Edge cases

| Case | Handling |
|---|---|
| Table-of-contents entries | A 10-Q lists its items before the body, so each anchor matches twice — on this corpus one risk-factors anchor hit 161 times across 39 documents. Sections shorter than `parse.min_section_chars` are dropped; TOC entries sit close together and fall under it, the real heading does not. |
| No anchor matches at all | One span covering the document, `section = parse.preamble_label` (null). Logged, counted in `(unsectioned)`, never dropped. |
| Text before the first anchor | Same treatment — kept with a null section. Rule 4: genuinely unattributed, and a guessed label would destroy that signal. |
| Exhibits | No Item structure at all (all 139 match no anchor). Their section is their manifest `exhibit_label`, which is EDGAR's own document type — sourced, not invented. Set `parse.exhibit_section_from_label: false` to treat them like any other document. |
| A form with no configured anchors | Falls back to `_default`. Lookup order is `<form>/<doc_type>`, then `<form>`, then `_default`. |
| Document produced no chunks | Logged with its accession and character count. |
| Provenance field is null | The chunk is **kept** and the null fields logged. A missing field is data about the source, not a defect to patch (Rule 4). |
| Manifest row never fetched | Counted as `not_fetched` and logged; stage 2 has work left. |
| Malformed markup | Text recovered so far is used, per `tools/htmltext.py`. |

## Rule notes

- **Rule 1**: reads what stage 2 left on disk; re-runnable in isolation.
- **Rule 2**: wall clock, items, bytes in/out, error count, zero requests.
- **Rule 5**: every chunk carries full provenance, enforced by the dataclass.
- **Rule 6**: the section distribution counts where text landed. It says nothing
  about what any filing says.
