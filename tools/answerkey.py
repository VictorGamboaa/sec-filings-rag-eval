"""Answer-key loading and ground-truth resolution, shared by verification and
evaluation.

WHY THIS IS ONE MODULE
======================

``tools.verify_key`` decides whether a key entry's quote is present in the
index. ``tools.evaluate`` decides whether retrieval returned the chunks that
quote identifies. Both questions rest on the same operation: given a ground
truth, which chunks satisfy it?

Two implementations of that would drift, and the drift would be silent -- the
key would verify against one matcher and be scored against another, so an entry
could pass verification and then be unscoreable, or worse, score against chunks
verification never blessed. One module, one answer.

MATCHING
========

Substring, after collapsing runs of whitespace on both sides. The collapse is
not fuzziness: it is the same normalization ``tools.htmltext`` already applies
when extracting text, so comparing raw key text against collapsed chunk text
would fail on formatting alone. Nothing else is normalized -- no case folding,
no punctuation substitution, no approximate matching. A near-match reported as a
match would manufacture agreement the key does not actually have.

SCOPING
=======

An accession identifies a FILING, and a filing owns its primary document and its
exhibits alike. A ground truth naming a ``document`` is scoped to that document,
so a match in an 8-K shell cannot satisfy a ground truth that lives in the
attached press release. ``accession: ANY`` opts out and searches the corpus.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

__all__ = [
    "AnswerKeyError",
    "normalize",
    "load_key",
    "ground_truths",
    "load_metadata",
    "MetadataIndex",
    "Resolution",
    "resolve",
    "index_fingerprint",
]

_WS = re.compile(r"\s+")


class AnswerKeyError(RuntimeError):
    """The key could not be read, or is not shaped as a key."""


def normalize(text: str) -> str:
    """Collapse whitespace. The only transformation applied to either side."""
    return _WS.sub(" ", str(text)).strip()


def load_key(path: str | Path) -> list[dict[str, Any]]:
    """Read the answer key. Never writes: the key is the record being checked."""
    import yaml

    path = Path(path)
    if not path.is_file():
        raise AnswerKeyError(
            f"{path} not found. Point evaluate.answer_key_path at the key, or "
            f"create it."
        )
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise AnswerKeyError(
            f"{path} must be a list of entries, got {type(data).__name__}."
        )
    return data


def ground_truths(entry: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize the three shapes ``ground_truth`` takes in the key.

    A mapping is one ground truth; a list is several (cross-company and
    period-over-period questions need more than one); ``null`` is a negative
    control, which is *supposed* to have none and is not a defect.
    """
    gt = entry.get("ground_truth")
    if gt is None:
        return []
    if isinstance(gt, dict):
        return [gt]
    if isinstance(gt, list):
        return [g for g in gt if isinstance(g, dict)]
    return []


def load_metadata(path: str | Path) -> list[dict[str, Any]]:
    """Read ``metadata.jsonl``. Line N describes FAISS row N."""
    path = Path(path)
    if not path.is_file():
        raise AnswerKeyError(
            f"{path} not found. Build the index first: python -m tools.embed_index"
        )
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


class MetadataIndex:
    """Lookups over the index metadata, prepared once instead of per quote.

    Normalizing 8,690 chunk texts for every ground truth would be the dominant
    cost of both tools; doing it once here keeps them linear in the key size.
    """

    def __init__(self, metadata: list[dict[str, Any]]) -> None:
        self.rows = metadata
        self.normalized: list[str] = [normalize(m["text"]) for m in metadata]
        self.by_accession: dict[str, list[int]] = {}
        for i, m in enumerate(metadata):
            self.by_accession.setdefault(m["accession"], []).append(i)
        # How many chunks share each exact text. A target whose text is
        # duplicated cannot be scored on chunk identity: its vector is identical
        # to its duplicates, so which one a search returns is row order, not
        # retrieval quality.
        self._text_counts: Counter[str] = Counter(self.normalized)
        self.by_chunk_id: dict[str, int] = {
            m["chunk_id"]: i for i, m in enumerate(metadata)
        }
        self._contains_cache: dict[str, list[int]] = {}

    def duplicate_text_count(self, row: int) -> int:
        return self._text_counts[self.normalized[row]]

    def rows_containing(self, needle: str) -> list[int]:
        """Every chunk containing this text, corpus-wide, ignoring scoping."""
        if needle not in self._contains_cache:
            self._contains_cache[needle] = [
                i for i, t in enumerate(self.normalized) if needle in t
            ]
        return self._contains_cache[needle]


class Resolution:
    """What a single ground truth resolves to, against one index."""

    __slots__ = (
        "status", "chunk_ids", "rows", "sections", "accession", "document",
        "ticker", "quote", "duplicate_text_count", "distinct_accessions",
        "distinct_texts", "corpus_chunks", "corpus_accessions",
    )

    def __init__(self, **kw: Any) -> None:
        for slot in self.__slots__:
            setattr(self, slot, kw.get(slot))

    @property
    def resolved(self) -> bool:
        return self.status in ("ok", "too_generic", "dedup_measured")

    def to_dict(self) -> dict[str, Any]:
        return {s: getattr(self, s) for s in self.__slots__}


def resolve(
    truth: dict[str, Any],
    index: MetadataIndex,
    *,
    generic_match_threshold: int = 3,
) -> Resolution:
    """Resolve one ground truth to the chunks that satisfy it.

    Statuses: ``ok``, ``too_generic`` (more matches than the threshold),
    ``dedup_measured`` (``accession: ANY`` -- breadth is the measurement, never
    a defect), ``no_quote``, ``accession_not_indexed``, ``document_not_indexed``,
    ``quote_absent``.
    """
    accession = truth.get("accession")
    document = truth.get("document")
    quote = truth.get("quote")
    common = {
        "accession": accession,
        "document": document,
        "ticker": truth.get("ticker"),
        "quote": quote,
        "chunk_ids": [],
        "rows": [],
        "sections": [],
    }

    if not quote:
        return Resolution(status="no_quote", **common)

    needle = normalize(quote)
    any_accession = str(accession).strip().upper() == "ANY"
    any_document = document is None or str(document).strip().upper() == "ANY"

    if any_accession:
        candidates = range(len(index.rows))
    else:
        candidates = index.by_accession.get(str(accession), [])
        if candidates and not any_document:
            candidates = [
                i for i in candidates if index.rows[i]["document"] == document
            ]

    rows = [i for i in candidates if needle in index.normalized[i]]

    if not rows:
        if any_accession:
            status = "quote_absent"
        elif not index.by_accession.get(str(accession)):
            status = "accession_not_indexed"
        elif not candidates:
            status = "document_not_indexed"
        else:
            status = "quote_absent"
        return Resolution(status=status, **common)

    common["chunk_ids"] = [index.rows[i]["chunk_id"] for i in rows]
    common["rows"] = list(rows)
    common["sections"] = sorted({str(index.rows[i]["section"]) for i in rows})

    # How widely the identifying text occurs CORPUS-WIDE, ignoring the scope.
    # This is what decides whether chunk-identity scoring is meaningful: if the
    # quote also appears in other filings, the embedder sees near-identical text
    # in several chunks and which of them ranks first turns on incidental
    # differences in surrounding words, not on retrieval quality.
    corpus_rows = index.rows_containing(needle)
    common["corpus_chunks"] = len(corpus_rows)
    common["corpus_accessions"] = len(
        {index.rows[i]["accession"] for i in corpus_rows}
    )

    if any_accession:
        status = "dedup_measured"
    elif len(rows) > generic_match_threshold:
        status = "too_generic"
    else:
        status = "ok"

    return Resolution(
        status=status,
        duplicate_text_count=max(index.duplicate_text_count(i) for i in rows),
        distinct_accessions=len({index.rows[i]["accession"] for i in rows}),
        distinct_texts=len({index.normalized[i] for i in rows}),
        **common,
    )


def index_fingerprint(sidecar: dict[str, Any]) -> str:
    """Identify the index a key was resolved against.

    Covers the embedder, the chunk fingerprint, the vector width, the metric and
    the COUNT. Count is included deliberately: a deduplication rebuild leaves the
    chunking config untouched, so the fingerprint would not move without it --
    and every chunk id the key resolves to would silently change meaning.
    """
    material = {
        "embedder": sidecar.get("embedder"),
        "chunk_fingerprint": sidecar.get("chunk_fingerprint"),
        "dim": sidecar.get("dim"),
        "metric": sidecar.get("metric"),
        "count": sidecar.get("count"),
    }
    import hashlib

    blob = json.dumps(material, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]
