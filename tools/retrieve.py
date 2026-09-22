"""Stage 5 -- retrieve: a query string -> top-k chunks from the FAISS index.

    python -m tools.retrieve "How much did Honeywell pay for Access Solutions?"
    python -m tools.retrieve "..." --top-k 5
    python -m tools.retrieve --query-file queries.txt
    python -m tools.retrieve "..." --set retrieve.top_k=50

Local files only once the model is cached, so the run log shows ``requests: 0``.

THE QUERY PREFIX IS NOT OPTIONAL
================================

bge models are trained with an asymmetric instruction: documents are embedded
bare, queries are embedded behind "Represent this sentence for searching
relevant passages: ". Embedding a query as though it were a document produces a
perfectly well-formed vector that is simply in the wrong place, and retrieval
degrades with nothing to show for it. That is why this stage calls
``embed_query`` and never ``embed_documents`` -- contract point 2 in
``tools.embedders.base``.

THE INDEX MUST MATCH THE EMBEDDER
=================================

Searching an index built by a different model returns results rather than an
error: the widths agree, FAISS is content, and the rankings are meaningless. So
the sidecar's embedder is compared against the constructed one and a mismatch
stops the run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from tools.answerkey import load_metadata
from tools.config import Config, ConfigError, add_config_args, load_config
from tools.embed_index import INDEX_FILE, METADATA_FILE, SIDECAR_FILE
from tools.embedders import EmbedderError, get_embedder
from tools.runlog import RunLog, StageRecorder

__all__ = ["Retriever", "retrieve"]


class Retriever:
    """An index, its metadata and a matching embedder, ready to search.

    Constructed once and reused: the evaluation harness runs 26 queries and
    reloading the model per query would dominate its timing.
    """

    def __init__(self, config: Config, stage: StageRecorder | None = None) -> None:
        import faiss

        self.config = config
        self.index_dir = Path(config.get("index.out_dir")) / str(config.get("index.name"))
        index_path = self.index_dir / INDEX_FILE
        if not index_path.is_file():
            raise ConfigError(
                f"{index_path} not found. Build it first: python -m tools.embed_index"
            )

        self.sidecar: dict[str, Any] = json.loads(
            (self.index_dir / SIDECAR_FILE).read_text(encoding="utf-8")
        )
        self.metadata = load_metadata(self.index_dir / METADATA_FILE)
        self.index = faiss.read_index(str(index_path))
        self.embedder = get_embedder(config, stage)

        if self.index.ntotal != len(self.metadata):
            raise ConfigError(
                f"index holds {self.index.ntotal} vectors but metadata has "
                f"{len(self.metadata)} rows; they are not row-aligned. "
                f"Re-run: python -m tools.embed_index"
            )
        self._check_embedder()

    def _check_embedder(self) -> None:
        built = self.sidecar.get("embedder") or {}
        now = self.embedder.describe()
        for field in ("model", "dim", "normalized"):
            if built.get(field) != now.get(field):
                raise EmbedderError(
                    f"index was built with embedder {field}={built.get(field)!r} "
                    f"but the configured embedder reports {now.get(field)!r}. "
                    f"Searching it would return well-formed, meaningless results. "
                    f"Point embed.provider/model at the index's embedder, or "
                    f"rebuild: python -m tools.embed_index --rebuild"
                )
        if self.index.d != self.embedder.dim:
            raise EmbedderError(
                f"index width {self.index.d} != embedder dim {self.embedder.dim}"
            )

    def search(
        self, query: str, top_k: int, stage: StageRecorder | None = None
    ) -> list[dict[str, Any]]:
        """Top-k hits for one query, best first."""
        import numpy as np

        vector = self.embedder.embed_query(query, stage)
        scores, ids = self.index.search(
            np.ascontiguousarray(vector.reshape(1, -1), dtype="float32"),
            min(top_k, self.index.ntotal),
        )
        hits = []
        for rank, (score, row) in enumerate(zip(scores[0], ids[0]), start=1):
            if row < 0:  # FAISS pads with -1 when fewer than k exist
                continue
            m = self.metadata[int(row)]
            hits.append(
                {
                    "rank": rank,
                    "score": round(float(score), 6),
                    "row": int(row),
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
        return hits


def _print_hits(query: str, hits: list[dict[str, Any]], width: int = 96) -> None:
    print(f"\n=== {query!r} ===")
    print(f"  {'#':>3} {'score':>8}  {'ticker':<6} {'accession':<22} "
          f"{'section':<34} document")
    for h in hits:
        print(f"  {h['rank']:>3} {h['score']:>8.4f}  {h['ticker']:<6} "
              f"{h['accession']:<22} {str(h['section'])[:34]:<34} {h['document']}")
    if hits:
        print("\n  top hit passage:")
        text = hits[0]["text"]
        for i in range(0, min(len(text), width * 4), width):
            print(f"    {text[i:i + width]}")
        if len(text) > width * 4:
            print("    ...")


def retrieve(
    config_path: str,
    overrides: list[str],
    queries: list[str],
    *,
    top_k: int | None = None,
    json_only: bool = False,
) -> int:
    config = load_config(config_path, overrides=overrides)
    k = int(top_k if top_k is not None else config.get("retrieve.top_k"))
    out_dir = Path(config.get("retrieve.out_dir"))

    results: list[dict[str, Any]] = []
    with RunLog(config) as log:
        with log.stage("retrieve") as stage:
            retriever = Retriever(config, stage)
            stage.note(
                index_dir=str(retriever.index_dir),
                index_size=retriever.index.ntotal,
                top_k=k,
                queries=len(queries),
            )
            for query in queries:
                hits = retriever.search(query, k, stage)
                results.append({"query": query, "top_k": k, "hits": hits})
                stage.count()
                if not json_only:
                    _print_hits(query, hits)

            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{log.run_id}.json"
            payload = json.dumps(
                {
                    "index_dir": str(retriever.index_dir),
                    "embedder": retriever.embedder.describe(),
                    "top_k": k,
                    "results": results,
                },
                indent=2,
                default=str,
            )
            out_path.write_text(payload, encoding="utf-8")
            stage.bytes_out(len(payload.encode("utf-8")))
            # No request() calls: the model is local and the index is on disk.
            stage.note(out_path=str(out_path))

    print(f"\nWrote {out_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tools.retrieve",
        description="Stage 5: search the index with a query string.",
    )
    add_config_args(parser)
    parser.add_argument("query", nargs="*", help="query string(s)")
    parser.add_argument(
        "--query-file", default=None,
        help="file of queries, one per line (blank lines and # comments ignored)",
    )
    parser.add_argument("--top-k", type=int, default=None, help="override retrieve.top_k")
    parser.add_argument("--json-only", action="store_true", help="suppress the table")
    args = parser.parse_args(argv)

    queries = list(args.query)
    if args.query_file:
        for line in Path(args.query_file).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                queries.append(line)
    if not queries:
        parser.error("give at least one query, or --query-file")

    try:
        return retrieve(
            args.config, args.overrides, queries,
            top_k=args.top_k, json_only=args.json_only,
        )
    except (ConfigError, EmbedderError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except ImportError as exc:
        print(f"ERROR: missing dependency ({exc}). Run: uv sync", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
