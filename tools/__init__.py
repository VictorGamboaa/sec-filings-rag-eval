"""Filing RAG evaluation harness — shared tooling.

Layer 3 of the WAT architecture: deterministic, testable Python. All concrete
actions live here. Workflows (markdown SOPs) drive these; the agent coordinates
but performs no external action directly.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any

__all__ = ["ChunkProvenance", "PROVENANCE_FIELDS"]


@dataclass(frozen=True, slots=True)
class ChunkProvenance:
    """Rule 5: every chunk carries full provenance.

    This is a hard schema rather than a loose dict so that no stage can persist
    a chunk with a field silently missing. Construction fails immediately if one
    is absent.

    Rule 4 interacts with this directly: a field whose value is genuinely
    unknown is ``None`` and must be logged as such. It is never back-filled,
    interpolated, or inferred -- a generated value has no source and no audit
    standing. ``None`` here means "the source did not provide this", which is a
    real finding; a plausible-looking guess destroys that signal.
    """

    cik: str
    ticker: str | None
    accession: str
    form: str
    filing_date: str | None
    period: str | None
    section: str | None
    chunk_ordinal: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def missing_fields(self) -> list[str]:
        """Names of provenance fields that are None.

        Stages should log this per chunk rather than discarding the chunk: a
        missing field is data about the source, not a defect to be patched.
        """
        return [f.name for f in fields(self) if getattr(self, f.name) is None]


PROVENANCE_FIELDS: tuple[str, ...] = tuple(f.name for f in fields(ChunkProvenance))
