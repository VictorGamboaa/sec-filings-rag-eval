"""Stage 4 -- embed+index: temp/chunks/ -> outputs/index/<name>/.

    python -m tools.embed_index
    python -m tools.embed_index --dry-run
    python -m tools.embed_index --limit 200
    python -m tools.embed_index --rebuild
    python -m tools.embed_index --set embed.batch_size=128

Batches chunks, embeds them through the configured embedder, and writes a FAISS
index with row-aligned metadata and a sidecar describing what produced it.

THE COMPATIBILITY GATE
======================

An index is only meaningful if every vector in it came from the same embedder
and the same chunking. Mixing two is not an error anyone would see: the vectors
have the same width, FAISS accepts them, search returns results, and the results
are quietly wrong. So the sidecar records ``embedder.describe()`` plus the chunk
config fingerprint, and a run whose fingerprint does not match is refused rather
than appended. ``--rebuild`` is the only way past it, and it starts a new index.

THE INDEX IS THE SOURCE OF TRUTH
================================

``index.faiss`` and ``metadata.jsonl`` must agree row for row, and they are
written by different mechanisms -- metadata is appended as it goes, the index is
serialized as a whole. A run killed between the two leaves metadata ahead of the
index, and resuming from that state would skip chunks that metadata claims are
present while every later row described the wrong vector. Search would return
confident, wrong provenance.

So the order is fixed: metadata is flushed first, the index written second, and
on load any metadata rows beyond ``index.ntotal`` are discarded and re-embedded.
Metadata ahead of the index is recoverable; the reverse would not be, since
nothing records which chunks those extra vectors came from.

OVER-LENGTH CHUNKS ARE REFUSED, NOT TRUNCATED
=============================================

Embedding backends truncate silently past ``max_seq_tokens`` -- a well-formed
vector comes back for the first N tokens with nothing indicating the rest was
dropped. Any chunk exceeding the embedder's declared limit is logged and skipped
instead. With the configured chunk size this should never fire; if it does, the
chunker and the embedder disagree about token counts, and that must be visible.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterator

from tools.config import Config, ConfigError, add_config_args, load_config
from tools.embedders import EmbedderError, get_embedder
from tools.parse_chunk import config_fingerprint
from tools.runlog import RunLog, StageRecorder
from tools.tokenize import TokenizerError, get_windower

__all__ = ["embed_index"]

INDEX_FILE = "index.faiss"
METADATA_FILE = "metadata.jsonl"
SIDECAR_FILE = "sidecar.json"

#: Fields kept in metadata.jsonl. The chunk text is kept so a retrieval result
#: can be read without going back to temp/, which is disposable by design.
METADATA_FIELDS: tuple[str, ...] = (
    "chunk_id",
    "cik",
    "ticker",
    "accession",
    "form",
    "filing_date",
    "period",
    "section",
    "chunk_ordinal",
    "doc_type",
    "document",
    "exhibit_label",
    "n_tokens",
    "char_span",
    "text",
)


def iter_chunks(chunk_dir: Path) -> Iterator[dict[str, Any]]:
    """Stream chunk records from every JSONL file, in a deterministic order.

    Sorted by filename so two runs over the same inputs assign the same FAISS
    row ids -- without that, an index rebuilt from identical data would not be
    comparable to its predecessor.
    """
    for path in sorted(chunk_dir.glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_no}: {exc}") from exc


def _read_sidecar(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _existing_ids(path: Path, committed: int) -> tuple[list[str], int]:
    """Chunk ids already IN THE INDEX, in row order, plus rows discarded.

    ``committed`` is the index's ``ntotal``. Metadata rows beyond it describe
    vectors that were never written -- the run was killed between the metadata
    flush and the index write -- so they are dropped and those chunks are
    embedded again. Trusting them instead would leave every subsequent row
    misaligned with the vector it is supposed to describe.
    """
    if not path.is_file():
        return [], 0
    rows: list[str] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line)["chunk_id"])
    if len(rows) <= committed:
        return rows, 0
    return rows[:committed], len(rows) - committed


def _truncate_metadata(path: Path, keep: int) -> None:
    """Rewrite metadata.jsonl to its first ``keep`` rows, atomically."""
    lines = [
        line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ][:keep]
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
    tmp.replace(path)


def _index_ntotal(path: Path) -> int:
    """Vectors already committed to disk. Zero when there is no index yet."""
    if not path.is_file():
        return 0
    import faiss

    return int(faiss.read_index(str(path)).ntotal)


def embed_index(
    config_path: str,
    overrides: list[str],
    *,
    dry_run: bool = False,
    limit: int | None = None,
    rebuild: bool = False,
) -> int:
    config = load_config(config_path, overrides=overrides)

    chunk_dir = Path(config.get("chunk.out_dir"))
    index_dir = Path(config.get("index.out_dir")) / str(config.get("index.name"))
    metric = str(config.get("index.metric", "ip")).lower()

    if not chunk_dir.is_dir():
        raise ConfigError(
            f"{chunk_dir} not found. Build chunks first: python -m tools.parse_chunk"
        )

    windower = get_windower(config)
    fingerprint = config_fingerprint(config, windower)

    chunks = list(iter_chunks(chunk_dir))
    if limit is not None:
        chunks = chunks[:limit]
    if not chunks:
        raise ConfigError(f"no chunk records found in {chunk_dir}")

    sidecar_path = index_dir / SIDECAR_FILE
    index_path = index_dir / INDEX_FILE
    metadata_path = index_dir / METADATA_FILE
    sidecar = None if rebuild else _read_sidecar(sidecar_path)

    # The index, not the metadata file, says what is actually committed.
    committed = 0 if rebuild else _index_ntotal(index_path)
    existing, orphaned = ([], 0) if rebuild else _existing_ids(metadata_path, committed)
    existing_set = set(existing)

    if sidecar is not None and sidecar.get("chunk_fingerprint") != fingerprint:
        raise ConfigError(
            f"{sidecar_path} was built from chunk config "
            f"{sidecar.get('chunk_fingerprint')!r}, but this run's chunks are "
            f"{fingerprint!r}. Appending would mix vectors from two different "
            f"chunkings in one index, which produces silently wrong rankings "
            f"rather than an error. Re-run with --rebuild to start a new index, "
            f"or set index.name to build alongside the existing one."
        )

    pending = [c for c in chunks if c["chunk_id"] not in existing_set]

    if dry_run:
        print("DRY RUN -- nothing embedded, nothing written.")
        print(f"  chunk dir        {chunk_dir} ({len(chunks):,} records)")
        print(f"  index dir        {index_dir}")
        print(f"  fingerprint      {fingerprint}")
        print(f"  sidecar          {'present' if sidecar else 'none'}"
              + (" (fingerprint matches)" if sidecar else ""))
        print(f"  index holds      {committed:,} vector(s)")
        if orphaned:
            print(f"  orphaned meta    {orphaned:,} row(s) ahead of the index "
                  f"(from an interrupted run) -- would be discarded and re-embedded")
        print(f"  would skip       {len(chunks) - len(pending):,} already embedded")
        print(f"  would embed      {len(pending):,} chunk(s)")
        print(f"  batch size       {config.get('embed.batch_size')}")
        print(f"  provider         {config.get('embed.provider')}")
        return 0

    import faiss
    import numpy as np

    embedded = 0
    over_length = 0

    with RunLog(config) as log:
        with log.stage("embed_index") as stage:
            embedder = get_embedder(config, stage)

            if metric == "ip" and not embedder.normalized:
                # Inner product equals cosine only for unit vectors. Building an
                # IP index over unnormalized vectors ranks by magnitude as much
                # as by direction, and nothing about the result says so.
                raise EmbedderError(
                    f"index.metric is 'ip' but {embedder!r} reports "
                    f"normalized=False. Inner product is only cosine similarity "
                    f"for normalized vectors; refusing to build a silently "
                    f"mis-ranked index."
                )
            if metric != "ip":
                raise ConfigError(
                    f"index.metric {metric!r} is not implemented; only 'ip' is."
                )

            stage.note(
                chunk_dir=str(chunk_dir),
                index_dir=str(index_dir),
                chunk_fingerprint=fingerprint,
                chunks_total=len(chunks),
                already_embedded=len(existing),
                batch_size=config.get("embed.batch_size"),
                rebuilt=rebuild,
            )

            # Refuse rather than truncate. See the module docstring.
            keep: list[dict[str, Any]] = []
            for chunk in pending:
                if int(chunk.get("n_tokens") or 0) > embedder.max_seq_tokens:
                    over_length += 1
                    stage.error(
                        "chunk exceeds the embedder's max_seq_tokens; skipped "
                        "rather than truncated",
                        context={
                            "chunk_id": chunk["chunk_id"],
                            "n_tokens": chunk.get("n_tokens"),
                            "max_seq_tokens": embedder.max_seq_tokens,
                        },
                    )
                    continue
                keep.append(chunk)

            index_dir.mkdir(parents=True, exist_ok=True)
            if not rebuild and index_path.is_file():
                index = faiss.read_index(str(index_path))
                if index.d != embedder.dim:
                    raise EmbedderError(
                        f"{index_path} has width {index.d}, embedder declares "
                        f"{embedder.dim}. Refusing to append."
                    )
            else:
                index = faiss.IndexFlatIP(embedder.dim)

            if orphaned:
                # Discard metadata describing vectors that were never written,
                # so appends land on a row boundary the index agrees with.
                _truncate_metadata(metadata_path, committed)
                stage.error(
                    "metadata rows ahead of the index were discarded; their "
                    "chunks are re-embedded. An earlier run was interrupted "
                    "between the metadata flush and the index write.",
                    context={"orphaned_rows": orphaned, "index_ntotal": committed},
                )
                stage.note(orphaned_metadata_rows=orphaned)

            batch_size = max(1, int(config.get("embed.batch_size", 64)))
            checkpoint_every = max(1, int(config.get("embed.checkpoint_batches", 20)))
            mode = "w" if (rebuild or not metadata_path.is_file()) else "a"
            checkpoints = 0

            def checkpoint(meta_fh: Any) -> None:
                """Commit metadata then the index, in that order.

                The order is the recovery guarantee: metadata ahead of the index
                is repairable on the next run by discarding the extra rows,
                whereas an index ahead of its metadata could not be -- nothing
                would record which chunks the extra vectors came from.
                """
                meta_fh.flush()
                os.fsync(meta_fh.fileno())
                faiss.write_index(index, str(index_path))

            with metadata_path.open(mode, encoding="utf-8") as meta_fh:
                for start in range(0, len(keep), batch_size):
                    batch = keep[start : start + batch_size]
                    vectors = embedder.embed_documents([c["text"] for c in batch], stage)
                    if vectors.shape[0] != len(batch):
                        raise EmbedderError(
                            f"embedder returned {vectors.shape[0]} vectors for "
                            f"{len(batch)} inputs; refusing to write a metadata "
                            f"file that would not be row-aligned with the index."
                        )
                    index.add(np.ascontiguousarray(vectors, dtype="float32"))
                    # Written after the add, in the same order, so metadata line
                    # N is FAISS row N.
                    for chunk in batch:
                        meta_fh.write(
                            json.dumps(
                                {k: chunk.get(k) for k in METADATA_FIELDS},
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                    embedded += len(batch)
                    if (start // batch_size + 1) % checkpoint_every == 0:
                        checkpoint(meta_fh)
                        checkpoints += 1
                checkpoint(meta_fh)

            stage.note(checkpoints=checkpoints)
            index_bytes = index_path.stat().st_size
            stage.bytes_out(index_bytes)

            sidecar_payload = {
                "embedder": embedder.describe(),
                "chunk_fingerprint": fingerprint,
                "windower": windower.identity,
                "dim": embedder.dim,
                "metric": metric,
                "count": index.ntotal,
                "index_file": INDEX_FILE,
                "metadata_file": METADATA_FILE,
            }
            sidecar_path.write_text(
                json.dumps(sidecar_payload, indent=2, default=str), encoding="utf-8"
            )

            if index.ntotal != len(existing) + embedded:
                stage.error(
                    "index count does not equal prior rows plus rows embedded",
                    context={
                        "ntotal": index.ntotal,
                        "existing": len(existing),
                        "embedded": embedded,
                    },
                )

            stage.note(
                embedded=embedded,
                skipped=len(chunks) - len(pending),
                over_length=over_length,
                dim=embedder.dim,
                metric=metric,
                index_size=index.ntotal,
                index_bytes=index_bytes,
            )
            ntotal = index.ntotal

    print(f"Indexed {embedded:,} chunk(s) into {index_dir}")
    print(f"  skipped {len(chunks) - len(pending):,} already embedded | "
          f"{over_length} over-length | index now holds {ntotal:,} vectors")
    print(f"  dim {embedder.dim} | metric {metric} | fingerprint {fingerprint}")
    if over_length:
        print(
            f"ERROR: {over_length} chunk(s) exceeded the embedder's token limit "
            f"and were not indexed.",
            file=sys.stderr,
        )
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tools.embed_index",
        description="Stage 4: embed chunks and build the FAISS index.",
    )
    add_config_args(parser)
    parser.add_argument(
        "--dry-run", action="store_true", help="report what would be embedded"
    )
    parser.add_argument(
        "--limit", type=int, default=None, metavar="N",
        help="only consider the first N chunk records",
    )
    parser.add_argument(
        "--rebuild", action="store_true",
        help="discard the existing index and build it again from scratch",
    )
    args = parser.parse_args(argv)

    try:
        return embed_index(
            args.config, args.overrides,
            dry_run=args.dry_run, limit=args.limit, rebuild=args.rebuild,
        )
    except (ConfigError, EmbedderError, TokenizerError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except ImportError as exc:
        print(f"ERROR: missing dependency ({exc}). Run: uv sync", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
