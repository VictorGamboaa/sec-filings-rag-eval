"""Hybrid retrieval: Reciprocal Rank Fusion over a dense and a lexical ranking.

RRF OPERATES ON RANKS
=====================

    score(d) = sum over retrievers of 1 / (k + rank_r(d))

Ranks, never scores. That is the reason to use it here: BM25 scores are
unnormalized and not comparable across queries, and cosine similarities sit on a
different scale entirely. Any attempt to combine the two by score requires
inventing a normalization, and an invented normalization is a free parameter
that quietly decides the result. Fusing ranks needs none.

k = 60 is the value from the original RRF paper and is not tuned here.

FULL-DEPTH FUSION
=================

Both rankings are fused at full depth rather than over a truncated candidate
list. Truncating would introduce a candidate-depth parameter -- how deep to take
each retriever before fusing -- which is another free choice that would need
justifying against the answer key. At this corpus size full depth is cheap, so
the parameter is removed rather than defended.

A document absent from a retriever's ranking contributes nothing from that
retriever, rather than being assigned a worst-case rank. Assigning one would be
a normalization decision of exactly the kind this avoids.
"""

from __future__ import annotations

from typing import Any

__all__ = ["RRF_K", "rrf_fuse", "HybridRetriever"]

RRF_K = 60


def rrf_fuse(
    rankings: list[list[int]], *, k: int = RRF_K, weights: list[float] | None = None
) -> list[tuple[int, float]]:
    """Fuse ranked lists of row ids. Returns ``(row, score)`` best first."""
    if weights is None:
        weights = [1.0] * len(rankings)
    scores: dict[int, float] = {}
    for ranking, weight in zip(rankings, weights):
        for rank, row in enumerate(ranking, start=1):
            scores[row] = scores.get(row, 0.0) + weight / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


class HybridRetriever:
    """Dense + BM25, fused by RRF. Same hit shape as the other retrievers."""

    kind = "hybrid"

    def __init__(self, dense: Any, lexical: Any, *, k: int = RRF_K) -> None:
        self.dense = dense
        self.lexical = lexical
        self.k = k
        self.metadata = dense.metadata
        self.sidecar = dense.sidecar
        self.binding_sidecar = getattr(dense, "binding_sidecar", dense.sidecar)
        self.index_dir = dense.index_dir
        if len(dense.metadata) != len(lexical.metadata):
            raise ValueError("dense and lexical retrievers disagree on chunk count")

    def _dense_ranking(self, query: str) -> list[int]:
        import numpy as np

        vector = self.dense.embedder.embed_query(query)
        n = self.dense.index.ntotal
        _, ids = self.dense.index.search(
            np.ascontiguousarray(vector.reshape(1, -1), dtype="float32"), n
        )
        return [int(r) for r in ids[0] if r >= 0]

    def search(self, query: str, top_k: int, stage: Any = None) -> list[dict[str, Any]]:
        fused = rrf_fuse(
            [self._dense_ranking(query), self.lexical.ranking(query)], k=self.k
        )
        hits = []
        for rank, (row, score) in enumerate(fused[:top_k], start=1):
            m = self.metadata[row]
            hits.append(
                {
                    "rank": rank,
                    "score": round(float(score), 8),
                    "row": row,
                    "chunk_id": m["chunk_id"],
                    "ticker": m["ticker"],
                    "accession": m["accession"],
                    "document": m["document"],
                    "doc_type": m["doc_type"],
                    "section": m["section"],
                    "form": m["form"],
                    "filing_date": m["filing_date"],
                    "text": m["text"],
                }
            )
        if stage is not None:
            stage.count()
        return hits
