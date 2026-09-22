# Workflow — Stage 5: retrieve

## Objective

Search the index with a query string and return the top-k chunks with full
provenance.

## Inputs

| Input | Source |
|---|---|
| `outputs/index/<name>/` | `workflows/04_embed_index.md` |
| `retrieve.top_k`, `retrieve.out_dir` | `inputs/config.yaml` |
| `embed.provider`, `embed.providers.local.*` | `inputs/config.yaml` |

## Tool

```
python -m tools.retrieve "How much did Honeywell pay for Access Solutions?"
python -m tools.retrieve "..." --top-k 5
python -m tools.retrieve --query-file queries.txt
```

**Runs in WSL at `~/dev/rag-test`,** like every stage. Local files only once the
model is cached, so the run log shows `requests: 0`.

## Outputs

| Output | Contents |
|---|---|
| stdout | Ranked table plus the top hit's passage |
| `outputs/retrievals/<run_id>.json` | Every hit with ticker, accession, document, section, chunk_id, score, text |
| `outputs/runs/<run_id>.jsonl` | wall clock, queries, bytes out, errors |

## The query prefix is not optional

bge models are trained asymmetrically: documents are embedded bare, queries
behind `Represent this sentence for searching relevant passages: `. This stage
calls **`embed_query`**, never `embed_documents`.

Embedding a query as a document does not fail. It returns a well-formed vector
that is simply in the wrong part of the space, and retrieval quietly gets worse
with nothing to show for it — contract point 2 in `tools/embedders/base.py`.

## The index must match the embedder

Searching an index built by a different model returns results rather than an
error: the widths agree, FAISS is content, the rankings are meaningless. So the
sidecar's `model`, `dim` and `normalized` are compared against the constructed
embedder and a mismatch **stops the run**.

The index and its metadata are also checked for row alignment — `ntotal` must
equal the metadata line count, or the provenance attached to every hit would be
wrong.

## Procedure

1. `python -m tools.retrieve "<question>"`.
2. Read the ranked table: score, ticker, accession, section, document.
3. Cross-check one hit against the key: `python -m tools.verify_key --only <id>`
   reports which chunk ids a ground truth resolves to.

## Edge cases

| Case | Handling |
|---|---|
| Index built by a different model | Refused before any search, naming both models. |
| `ntotal` != metadata rows | Refused; the index and metadata are not row-aligned. |
| `top_k` larger than the index | Clamped to `ntotal`; FAISS's `-1` padding rows are dropped. |
| Model not cached | First run downloads it (~130 MB). Network; ask first. |
| No index yet | Names the stage that builds it. |

## Rule notes

- **Rule 2**: wall clock, query count, bytes out, `requests: 0`.
- **Rule 3**: `top_k`, the index name and the embedder are all config.
- **Rule 6**: the stage returns passages and scores. It does not summarize or
  interpret what the passages say.
