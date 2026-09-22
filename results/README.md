# Results

Machine-readable output behind the tables in the top-level README.

## eval/

One evaluation run per configuration, named `<retriever>_<index>.json`.

| file | configuration |
|---|---|
| dense_main.json | bge-small, 450 tokens |
| dense_main_200.json | bge-small, 200 tokens |
| dense_main_base.json | bge-base, 450 tokens |
| dense_main_voyage.json | voyage-finance-2, 450 tokens |
| bm25_main_bm25.json | BM25, 450-token chunks |
| hybrid_main.json | bge-small + BM25, RRF k=60 |
| hybrid_main_voyage.json | voyage + BM25, RRF k=60 |

Each file records `retriever`, `index_dir`, `index_fingerprint`, `top_k`,
and the answer key path, so a result can be traced to the index that
produced it.

`dense_main_200.json` predates the `retriever` field being written to the
output, so its retriever is recorded as unknown in the file itself. It is
dense by construction: the BM25 index did not exist when that run was made.

## keyverify/

Answer key verification per index. Every ground-truth quote is checked
verbatim against the indexed chunks before any scoring runs.

## sidecars/

Build provenance per index: model, dimension, chunk fingerprint, and for
the voyage build the billed token and request counts.
