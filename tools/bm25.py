"""BM25 lexical retrieval over the same chunks as the dense index.

Built because dense cosine similarity averages away rare terms. A corpus of SEC
filings is mostly boilerplate: thousands of chunks differ only in a company
name, a coupon rate or a date. Those rare tokens -- "Sundyne", "Corebridge",
"6.250%" -- are exactly what the single-fact questions ask about, and they are
what a lexical score keys on.

NO NEW DEPENDENCY
=================

BM25 is implemented here rather than pulled in. It is forty lines of arithmetic,
and writing it keeps k1/b explicit and auditable rather than buried in a
library's defaults.

PARAMETERS ARE THE STANDARD ONES AND ARE NOT TUNED
==================================================

k1=1.2, b=0.75 -- the Robertson/Lucene defaults. Tuning them against the answer
key would produce a number that describes the key rather than the method, and
could not be reported as a measurement of anything.

TOKENIZATION
============

Lowercase, split on non-alphanumerics, no stemming, no stopword removal -- and
numerics survive intact. A naive split on non-alphanumerics would shatter
"6.250%" into "6" and "250", destroying precisely the tokens that make a chunk
identifiable. So a number keeps its internal separators and a trailing percent
sign: "6.250%", "48.4", "4,913" each stay one token.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

__all__ = [
    "TOKEN_RE",
    "tokenize",
    "BM25Index",
    "BM25Retriever",
    "BM25_FILE",
    "SIDECAR_FILE",
]

BM25_FILE = "bm25.json"
SIDECAR_FILE = "sidecar.json"

#: A token is a run of alphanumerics, optionally continued through internal
#: "." or "," into further alphanumerics, optionally ending in "%".
#: "6.250%" -> one token; "48.4" -> one token; "4,913" -> one token.
TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.,][a-z0-9]+)*%?")

DEFAULT_K1 = 1.2
DEFAULT_B = 0.75


def tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphanumerics, keep numerics whole."""
    return TOKEN_RE.findall(str(text).lower())


class BM25Index:
    """An inverted index with Robertson/Lucene BM25 scoring."""

    def __init__(
        self,
        doc_ids: list[str],
        doc_len: list[int],
        postings: dict[str, list[tuple[int, int]]],
        *,
        k1: float = DEFAULT_K1,
        b: float = DEFAULT_B,
    ) -> None:
        self.doc_ids = doc_ids
        self.doc_len = doc_len
        self.postings = postings
        self.k1 = float(k1)
        self.b = float(b)
        self.n_docs = len(doc_ids)
        self.avgdl = (sum(doc_len) / self.n_docs) if self.n_docs else 0.0
        # Robertson/Sparck-Jones IDF with the +1 that keeps it non-negative.
        self.idf = {
            term: math.log(
                1.0 + (self.n_docs - len(post) + 0.5) / (len(post) + 0.5)
            )
            for term, post in postings.items()
        }

    # -- build / persist ---------------------------------------------------

    @classmethod
    def build(
        cls,
        docs: Iterable[tuple[str, str]],
        *,
        k1: float = DEFAULT_K1,
        b: float = DEFAULT_B,
    ) -> BM25Index:
        """Build from ``(chunk_id, text)`` pairs, in the order given.

        Order is preserved so a BM25 row index maps to the same chunk as the
        dense index's row, which is what lets the two rankings be fused.
        """
        doc_ids: list[str] = []
        doc_len: list[int] = []
        postings: dict[str, list[tuple[int, int]]] = {}
        for i, (chunk_id, text) in enumerate(docs):
            tokens = tokenize(text)
            doc_ids.append(chunk_id)
            doc_len.append(len(tokens))
            for term, tf in Counter(tokens).items():
                postings.setdefault(term, []).append((i, tf))
        return cls(doc_ids, doc_len, postings, k1=k1, b=b)

    def save(self, path: Path) -> int:
        payload = {
            "k1": self.k1,
            "b": self.b,
            "doc_ids": self.doc_ids,
            "doc_len": self.doc_len,
            "postings": {t: [list(p) for p in post]
                         for t, post in self.postings.items()},
        }
        blob = json.dumps(payload)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(blob, encoding="utf-8")
        tmp.replace(path)
        return len(blob.encode("utf-8"))

    @classmethod
    def load(cls, path: Path) -> BM25Index:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            payload["doc_ids"],
            payload["doc_len"],
            {t: [tuple(p) for p in post]
             for t, post in payload["postings"].items()},
            k1=payload["k1"],
            b=payload["b"],
        )

    # -- search ------------------------------------------------------------

    def scores(self, query: str) -> dict[int, float]:
        """BM25 score per document, for documents matching at least one term.

        Documents matching no query term are absent rather than scored zero --
        "no lexical evidence at all" is a different statement from "scored
        lowest", and the negative-control analysis depends on the difference.
        """
        out: dict[int, float] = {}
        for term, qtf in Counter(tokenize(query)).items():
            post = self.postings.get(term)
            if not post:
                continue
            idf = self.idf[term]
            for doc, tf in post:
                denom = tf + self.k1 * (
                    1.0 - self.b + self.b * self.doc_len[doc] / self.avgdl
                )
                out[doc] = out.get(doc, 0.0) + idf * (tf * (self.k1 + 1.0)) / denom
        return out

    def search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        ranked = sorted(self.scores(query).items(), key=lambda kv: (-kv[1], kv[0]))
        return ranked[:top_k]

    def ranking(self, query: str) -> list[int]:
        """Full ranking of matching documents, best first."""
        return [doc for doc, _ in
                sorted(self.scores(query).items(), key=lambda kv: (-kv[1], kv[0]))]


class BM25Retriever:
    """Search interface matching tools.retrieve.Retriever.

    Exposes the same hit shape so tools.evaluate can score dense, lexical and
    fused runs with one scoring implementation rather than three.
    """

    kind = "bm25"

    def __init__(self, index_dir: Path, metadata: list[dict[str, Any]],
                 binding_sidecar: dict[str, Any] | None = None) -> None:
        self.index_dir = Path(index_dir)
        self.metadata = metadata
        self.sidecar = json.loads(
            (self.index_dir / SIDECAR_FILE).read_text(encoding="utf-8")
        )
        self.bm25 = BM25Index.load(self.index_dir / BM25_FILE)
        # The binding check asks whether the key's chunk ids are valid here.
        # They are iff the chunk set is identical, which the build asserts and
        # the sidecar records; the dense sidecar is carried so evaluate can
        # compare against the record verify_key wrote for that same chunk set.
        self.binding_sidecar = binding_sidecar or self.sidecar
        if len(self.bm25.doc_ids) != len(metadata):
            raise ValueError(
                f"bm25 holds {len(self.bm25.doc_ids)} docs but metadata has "
                f"{len(metadata)} rows"
            )
        self.row_of = {cid: i for i, cid in enumerate(self.bm25.doc_ids)}

    def search(self, query: str, top_k: int, stage: Any = None) -> list[dict[str, Any]]:
        hits = []
        for rank, (row, score) in enumerate(self.bm25.search(query, top_k), start=1):
            m = self.metadata[row]
            hits.append(
                {
                    "rank": rank,
                    "score": round(float(score), 6),
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

    def ranking(self, query: str) -> list[int]:
        return self.bm25.ranking(query)
