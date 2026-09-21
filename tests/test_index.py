"""Tests for stage 4: the compatibility gate, row alignment, and skipping.

Uses a stub embedder so the suite needs no model download. The stub honours the
same contract the real one does -- declared dim, declared normalization,
declared token limit -- which is the point of having the contract.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import tools.embed_index as ei
from tools.config import load_config
from tools.embed_index import METADATA_FILE, SIDECAR_FILE, embed_index, iter_chunks


class StubEmbedder:
    """Deterministic, normalized, 8-dimensional."""

    name = "stub"
    dim = 8
    max_batch = 4
    normalized = True
    max_seq_tokens = 100

    def __init__(self) -> None:
        self.calls = 0

    def _vec(self, text: str) -> np.ndarray:
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        v = rng.standard_normal(self.dim).astype("float32")
        return v / np.linalg.norm(v)

    def embed_documents(self, texts, stage=None):
        self.calls += 1
        out = np.vstack([self._vec(t) for t in texts]).astype("float32")
        if stage is not None:
            stage.count(len(texts))
            stage.bytes_out(int(out.nbytes))
        return out

    def embed_query(self, text, stage=None):
        return self._vec(text)

    def describe(self):
        return {
            "provider": self.name,
            "model": "stub-v1",
            "dim": self.dim,
            "normalized": self.normalized,
            "max_seq_tokens": self.max_seq_tokens,
        }


@pytest.fixture
def stub(monkeypatch):
    embedder = StubEmbedder()
    monkeypatch.setattr(ei, "get_embedder", lambda config, stage=None: embedder)
    return embedder


def _chunk(i: int, n_tokens: int = 10, **kw):
    base = {
        "chunk_id": f"0000000001-26-00000{i // 10}:doc.htm:{i:05d}",
        "cik": "0000000001",
        "ticker": "TEST",
        "accession": "0000000001-26-000001",
        "form": "8-K",
        "filing_date": "2026-01-02",
        "period": "2026-01-01",
        "section": "item_2.02",
        "chunk_ordinal": i,
        "doc_type": "primary",
        "document": "doc.htm",
        "exhibit_label": None,
        "n_tokens": n_tokens,
        "char_span": [i * 10, i * 10 + 10],
        "text": f"chunk number {i} body text",
    }
    base.update(kw)
    return base


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """A chunk dir plus config overrides pointing the stage at tmp_path."""
    monkeypatch.chdir(tmp_path)
    chunk_dir = tmp_path / "chunks"
    chunk_dir.mkdir()
    index_dir = tmp_path / "index"

    def write(chunks, name="a.jsonl"):
        (chunk_dir / name).write_text(
            "".join(json.dumps(c) + "\n" for c in chunks), encoding="utf-8"
        )

    overrides = [
        f"chunk.out_dir={chunk_dir}",
        f"index.out_dir={index_dir}",
        "index.name=test",
        f"run.log_dir={tmp_path / 'runs'}",
        "chunk.unit=chars",
        "embed.batch_size=3",
    ]
    return write, overrides, index_dir / "test"


def _config_path() -> str:
    return str((__import__("pathlib").Path(__file__).parent.parent / "inputs" / "config.yaml"))


def _run(overrides, **kw):
    return embed_index(_config_path(), overrides, **kw)


# --------------------------------------------------------------------------
# Streaming chunks
# --------------------------------------------------------------------------


def test_iter_chunks_is_deterministic_across_files(tmp_path):
    (tmp_path / "b.jsonl").write_text(json.dumps(_chunk(2)) + "\n", encoding="utf-8")
    (tmp_path / "a.jsonl").write_text(json.dumps(_chunk(1)) + "\n", encoding="utf-8")
    ids = [c["chunk_ordinal"] for c in iter_chunks(tmp_path)]
    assert ids == [1, 2], "file order must be sorted so row ids are reproducible"


def test_iter_chunks_skips_blank_lines(tmp_path):
    (tmp_path / "a.jsonl").write_text(
        json.dumps(_chunk(1)) + "\n\n" + json.dumps(_chunk(2)) + "\n", encoding="utf-8"
    )
    assert len(list(iter_chunks(tmp_path))) == 2


def test_iter_chunks_names_the_file_and_line_on_bad_json(tmp_path):
    (tmp_path / "a.jsonl").write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="a.jsonl:1"):
        list(iter_chunks(tmp_path))


# --------------------------------------------------------------------------
# Building, row alignment, resuming
# --------------------------------------------------------------------------


def test_builds_an_index_with_row_aligned_metadata(workspace, stub):
    write, overrides, index_dir = workspace
    chunks = [_chunk(i) for i in range(7)]
    write(chunks)

    assert _run(overrides) == 0

    import faiss

    index = faiss.read_index(str(index_dir / "index.faiss"))
    meta = [
        json.loads(line)
        for line in (index_dir / METADATA_FILE).read_text(encoding="utf-8").splitlines()
    ]
    assert index.ntotal == 7
    assert len(meta) == 7
    assert [m["chunk_ordinal"] for m in meta] == list(range(7))

    # Row alignment is the property that matters: searching for a chunk's own
    # vector must return the metadata row describing that chunk.
    for row in (0, 3, 6):
        vec = stub._vec(meta[row]["text"]).reshape(1, -1)
        _, ids = index.search(vec, 1)
        assert ids[0][0] == row, "FAISS row and metadata line have diverged"


def test_sidecar_records_what_produced_the_index(workspace, stub):
    write, overrides, index_dir = workspace
    write([_chunk(i) for i in range(3)])
    _run(overrides)
    sidecar = json.loads((index_dir / SIDECAR_FILE).read_text(encoding="utf-8"))
    assert sidecar["embedder"]["model"] == "stub-v1"
    assert sidecar["dim"] == 8
    assert sidecar["metric"] == "ip"
    assert sidecar["count"] == 3
    assert sidecar["chunk_fingerprint"]


def test_rerun_embeds_nothing_and_leaves_the_index_unchanged(workspace, stub):
    write, overrides, index_dir = workspace
    write([_chunk(i) for i in range(5)])
    _run(overrides)
    calls_after_first = stub.calls

    assert _run(overrides) == 0
    assert stub.calls == calls_after_first, "a re-run must not re-embed"

    import faiss

    assert faiss.read_index(str(index_dir / "index.faiss")).ntotal == 5


def test_new_chunks_append_without_re_embedding_the_old(workspace, stub):
    write, overrides, index_dir = workspace
    write([_chunk(i) for i in range(4)])
    _run(overrides)
    first_calls = stub.calls

    write([_chunk(i) for i in range(4)] + [_chunk(i) for i in range(4, 6)])
    assert _run(overrides) == 0

    import faiss

    index = faiss.read_index(str(index_dir / "index.faiss"))
    meta = (index_dir / METADATA_FILE).read_text(encoding="utf-8").splitlines()
    assert index.ntotal == 6
    assert len(meta) == 6
    assert stub.calls > first_calls
    # Only the two new chunks were embedded: batch size 3 means one extra call.
    assert stub.calls == first_calls + 1


def test_batching_respects_embed_batch_size(workspace, stub):
    write, overrides, _ = workspace
    write([_chunk(i) for i in range(7)])
    _run(overrides)
    assert stub.calls == 3, "7 chunks at batch size 3 is 3 calls"


# --------------------------------------------------------------------------
# The compatibility gate
# --------------------------------------------------------------------------


def test_mismatched_chunk_fingerprint_is_refused(workspace, stub):
    write, overrides, index_dir = workspace
    write([_chunk(i) for i in range(3)])
    _run(overrides)

    # Changing a chunk tunable changes the fingerprint.
    changed = overrides + ["chunk.size=123"]
    with pytest.raises(Exception) as exc:
        _run(changed)
    assert "--rebuild" in str(exc.value), (
        "the refusal must name the way forward, since mixing chunkings in one "
        "index degrades ranking silently rather than failing"
    )


def test_rebuild_starts_a_fresh_index(workspace, stub):
    write, overrides, index_dir = workspace
    write([_chunk(i) for i in range(5)])
    _run(overrides)

    write([_chunk(i) for i in range(2)])
    assert _run(overrides + ["chunk.size=123"], rebuild=True) == 0

    import faiss

    assert faiss.read_index(str(index_dir / "index.faiss")).ntotal == 2
    meta = (index_dir / METADATA_FILE).read_text(encoding="utf-8").splitlines()
    assert len(meta) == 2, "metadata must be rewritten, not appended to"


def test_unnormalized_embedder_is_refused_for_an_ip_index(workspace, monkeypatch):
    write, overrides, _ = workspace
    write([_chunk(i) for i in range(2)])

    class Unnormalized(StubEmbedder):
        normalized = False

    monkeypatch.setattr(ei, "get_embedder", lambda config, stage=None: Unnormalized())
    with pytest.raises(Exception, match="normalized"):
        _run(overrides)


# --------------------------------------------------------------------------
# Over-length chunks
# --------------------------------------------------------------------------


def test_over_length_chunk_is_skipped_not_truncated(workspace, stub):
    write, overrides, index_dir = workspace
    write([_chunk(0), _chunk(1, n_tokens=5000), _chunk(2)])

    assert _run(overrides) == 1, "an over-length chunk must make the stage fail loudly"

    import faiss

    index = faiss.read_index(str(index_dir / "index.faiss"))
    assert index.ntotal == 2, "the over-length chunk must not be in the index"
    meta = [
        json.loads(line)
        for line in (index_dir / METADATA_FILE).read_text(encoding="utf-8").splitlines()
    ]
    assert [m["chunk_ordinal"] for m in meta] == [0, 2]


def test_chunk_exactly_at_the_limit_is_kept(workspace, stub):
    write, overrides, index_dir = workspace
    write([_chunk(0, n_tokens=StubEmbedder.max_seq_tokens)])
    assert _run(overrides) == 0

    import faiss

    assert faiss.read_index(str(index_dir / "index.faiss")).ntotal == 1


# --------------------------------------------------------------------------
# Dry run
# --------------------------------------------------------------------------


def test_dry_run_writes_nothing(workspace, stub):
    write, overrides, index_dir = workspace
    write([_chunk(i) for i in range(3)])
    assert _run(overrides, dry_run=True) == 0
    assert not index_dir.exists()
    assert stub.calls == 0


# --------------------------------------------------------------------------
# Crash recovery: the index is the source of truth
# --------------------------------------------------------------------------


def test_metadata_ahead_of_the_index_is_discarded_and_re_embedded(workspace, stub):
    """The defect found by an interrupted real build.

    Metadata is appended per batch; the index is serialized separately. A run
    killed between the two leaves metadata rows describing vectors that were
    never written. Resuming naively would skip those chunks forever and leave
    every later row misaligned with its vector -- search returning confident,
    wrong provenance.
    """
    import faiss

    write, overrides, index_dir = workspace
    chunks = [_chunk(i) for i in range(6)]
    write(chunks)
    _run(overrides)

    # Simulate the interrupted state: metadata ahead of the index.
    index = faiss.read_index(str(index_dir / "index.faiss"))
    assert index.ntotal == 6
    truncated = faiss.IndexFlatIP(index.d)
    truncated.add(index.reconstruct_n(0, 2))
    faiss.write_index(truncated, str(index_dir / "index.faiss"))
    assert len(
        (index_dir / METADATA_FILE).read_text(encoding="utf-8").splitlines()
    ) == 6

    assert _run(overrides) == 0

    recovered = faiss.read_index(str(index_dir / "index.faiss"))
    meta = [
        json.loads(line)
        for line in (index_dir / METADATA_FILE).read_text(encoding="utf-8").splitlines()
    ]
    assert recovered.ntotal == 6
    assert len(meta) == 6, "orphaned rows must be discarded, not appended past"
    assert [m["chunk_ordinal"] for m in meta] == list(range(6))

    # Alignment restored: every row's own vector retrieves that row.
    for row in (0, 2, 5):
        vec = stub._vec(meta[row]["text"]).reshape(1, -1)
        _, ids = recovered.search(vec, 1)
        assert ids[0][0] == row


def test_checkpoint_commits_the_index_partway_through(workspace, stub):
    """A long build must survive interruption with work retained."""
    import faiss

    write, overrides, index_dir = workspace
    write([_chunk(i) for i in range(9)])
    # batch_size 3, checkpoint every batch => index written as it goes.
    _run(overrides + ["embed.checkpoint_batches=1"])
    assert faiss.read_index(str(index_dir / "index.faiss")).ntotal == 9


def test_index_ahead_of_metadata_is_not_produced(workspace, stub):
    """Metadata is flushed before the index, never after.

    The reverse order would be unrecoverable: nothing records which chunks the
    surplus vectors came from.
    """
    write, overrides, index_dir = workspace
    write([_chunk(i) for i in range(7)])
    _run(overrides + ["embed.checkpoint_batches=1"])

    import faiss

    ntotal = faiss.read_index(str(index_dir / "index.faiss")).ntotal
    rows = len((index_dir / METADATA_FILE).read_text(encoding="utf-8").splitlines())
    assert rows >= ntotal, "the index must never run ahead of its metadata"
    assert rows == ntotal
