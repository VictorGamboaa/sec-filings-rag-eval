# Workflow — Stage 4: embed + index

## Objective

Embed the chunks and build a FAISS index with row-aligned metadata and a sidecar
recording exactly what produced it.

## Inputs

| Input | Source |
|---|---|
| `temp/chunks/*.jsonl` | `workflows/03_parse_chunk.md` |
| `embed.provider`, `embed.batch_size`, `embed.providers.local.*` | `inputs/config.yaml` |
| `index.out_dir`, `index.name`, `index.metric` | `inputs/config.yaml` |

## Tool

```
python -m tools.embed_index [--dry-run] [--limit N] [--rebuild] [--set k=v ...]
```

**Must run in WSL at `~/dev/rag-test`.** On the Windows host
`sentence_transformers` cannot import at all — scipy's compiled extensions are
blocked by an Application Control policy, which `tools/embedders/local.py`
diagnoses by name. Nothing about that is fixable by reinstalling.

The first run downloads the model (~130 MB) from HuggingFace. That is an external
fetch: **ask before running it.** Later runs use the local cache.

## Outputs

| Output | Contents |
|---|---|
| `outputs/index/<name>/index.faiss` | The vectors |
| `outputs/index/<name>/metadata.jsonl` | One line per vector, **row-aligned with the FAISS ids** |
| `outputs/index/<name>/sidecar.json` | `embedder.describe()`, chunk fingerprint, dim, metric, count |
| `outputs/runs/<run_id>.jsonl` | wall clock, chunks embedded, bytes out, errors |

The index directory name comes from `index.name`, not the run id, so a partial
build resumes rather than starting a new index each run. The sidecar is what
decides whether resuming is legitimate.

## The compatibility gate

An index is only meaningful if every vector in it came from the same embedder and
the same chunking. Mixing two is **not an error anyone would see**: the vectors
have the same width, FAISS accepts them, search returns results, and the results
are quietly wrong.

So a run whose chunk fingerprint differs from the sidecar's is **refused**, and
the refusal names `--rebuild`. Use `--rebuild` to start a new index, or set
`index.name` to build alongside the existing one for comparison.

`IndexFlatIP` is chosen only after asserting `embedder.normalized` — inner product
equals cosine similarity only for unit vectors. An IP index over unnormalized
vectors ranks partly by magnitude, and nothing in the output says so.

## Over-length chunks are refused, never truncated

Embedding backends truncate silently past `max_seq_tokens`. Any chunk exceeding
the embedder's declared limit is logged with its id, skipped, and the stage
**exits non-zero**.

With `chunk.size: 450` against this model's 512 this should never fire. If it
does, the chunker and the embedder disagree about token counts — which is a real
defect, and the whole point of making it loud.

## Crash recovery: the index is the source of truth

`index.faiss` and `metadata.jsonl` must agree row for row, and they are written
by different mechanisms — metadata is appended as it goes, the index is
serialized as a whole. **A run killed between the two leaves metadata ahead of
the index.** Found the hard way on a real build: an interrupted run left 200
vectors against 5,256 metadata rows.

Resuming naively from that state would skip the 5,056 chunks metadata claims are
present, and every row past 200 would describe a different chunk than the FAISS
row it is aligned to. Search would return confident, wrong provenance — no error
anywhere.

So:

- The **index's `ntotal`** decides what is committed, not the metadata file.
- On load, metadata rows beyond it are discarded and those chunks re-embedded.
  The stage logs this and reports `orphaned_metadata_rows`.
- Within a run, metadata is flushed and fsynced **before** the index is written.
  That order is the recovery guarantee: metadata ahead of the index is
  repairable, whereas an index ahead of its metadata would not be — nothing
  records which chunks the surplus vectors came from.
- `embed.checkpoint_batches` commits both every N batches, so a long CPU build
  that is interrupted resumes from the last checkpoint rather than restarting.

## Skip behaviour

Chunk ids already present in `metadata.jsonl` **and backed by a committed
vector** are skipped; the rest are embedded and appended. `IndexFlatIP` supports
incremental `add`, so a partial build resumes instead of restarting.

Chunk streaming is ordered by filename so two runs over identical inputs assign
identical FAISS row ids — without that an index rebuilt from the same data would
not be comparable to its predecessor.

## Procedure

1. Read stage 3's section distribution first. An index built on bad sectioning
   costs a full re-embed to correct.
2. `python -m tools.embed_index --dry-run` — confirm counts and fingerprint.
3. Ask for approval, then `python -m tools.embed_index --limit 200`. Confirm
   `dim` 384, `over_length: 0`, and that the sidecar names the embedder.
4. Ask for approval, then the full run.
5. Re-run and confirm `embedded: 0` with everything skipped.
6. Sanity retrieval: embed a query with the bge query prefix, search, and confirm
   the returned row id indexes the metadata line describing that chunk.

## Edge cases

| Case | Handling |
|---|---|
| Sidecar fingerprint differs | Refused, naming `--rebuild`. Never appended. |
| Existing index width ≠ embedder `dim` | Refused before any vector is added. |
| `index.metric` is not `ip` | Refused; only inner product is implemented. |
| Embedder returns fewer vectors than inputs | Refused — writing that metadata would break row alignment, which nothing downstream could detect. |
| Chunk over `max_seq_tokens` | Logged, skipped, non-zero exit. |
| Interrupted mid-run | Work up to the last checkpoint is kept. Metadata rows beyond the index's `ntotal` are discarded on the next run and re-embedded; `orphaned_metadata_rows` records how many. |
| `sentence_transformers` will not import | `tools/embedders/local.py::_diagnose_import` distinguishes "not installed" from "installed but a native extension is blocked" — the second is a host policy, and reinstalling will not help. |
| Model not cached | First run downloads it. Network; ask first. |

## Rule notes

- **Rule 2**: the local embedder reports **zero requests and zero tokens** by
  contract. That zero is load-bearing: it is what makes a later local-vs-hosted
  cost and throughput comparison legible.
- **Rule 3**: batch size, metric, index name and the model are all config.
- **Rule 4**: a chunk that cannot be embedded is absent and logged. No zero
  vector, no mean vector, no substitute is ever written — a fabricated embedding
  would pollute every retrieval metric computed downstream.
