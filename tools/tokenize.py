"""Tokenization for the chunker: exact counts and exact character spans.

``chunk.unit`` selects the unit the chunker measures in. Two are implemented:

``tokens``
    The tokenizer of the configured embedding model, via HuggingFace. Counts are
    then the same counts the model will apply, which is the only way the
    chunker's ``chunk.size`` can be validated against the embedder's
    ``max_seq_tokens``. A generic approximation -- words times some factor --
    would put chunks over the model limit while reporting them under it, and
    the truncation that followed would be silent.

``chars``
    Character windows. No model, no download. Useful for exercising the pipeline
    where the model is unavailable, and for isolating tokenizer cost when
    measuring throughput.

WHY OFFSETS
===========

Both units return windows as ``(start_char, end_char)`` into the text they were
given, plus the token count for that window. Spans come from the tokenizer's own
offset mapping rather than from re-joining decoded tokens, because decoding is
lossy: BERT WordPiece drops the distinction between "##ing" attached to the
previous word and a separate token, and re-joining would produce text that does
not appear in the source document. A chunk whose text cannot be located in the
document it claims to come from has no provenance, which Rule 5 does not allow.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:  # pragma: no cover
    from tools.config import Config

__all__ = [
    "TokenizerError",
    "Windower",
    "get_windower",
    "CharWindower",
    "TokenWindower",
]


class TokenizerError(RuntimeError):
    """A tokenizer could not be constructed, or was asked for a bad window."""


#: Sentence terminator followed by whitespace. Deliberately simple: the cost of
#: a wrong split here is a chunk boundary one sentence off, not lost text.
_SENTENCE_END = re.compile(r"[.!?][\"')\]]*\s")


@runtime_checkable
class Windower(Protocol):
    """Splits text into overlapping windows of a configured size."""

    #: Identifies the unit and, for tokens, the exact tokenizer. Part of the
    #: chunk config fingerprint: token counts are meaningless without it.
    identity: str

    def count(self, text: str) -> int:
        """Length of ``text`` in this windower's unit."""
        ...

    def windows(self, text: str) -> list[tuple[int, int, int]]:
        """Return ``(start_char, end_char, n_units)`` covering ``text``."""
        ...


def _snap_end(text: str, start: int, end: int, fraction: float) -> int:
    """Move ``end`` back to a sentence boundary inside the last ``fraction``.

    Returns ``end`` unchanged when no boundary is found, or when snapping would
    leave a window so short that progress stalls.
    """
    if end >= len(text) or fraction <= 0:
        return end
    window = end - start
    earliest = max(start + 1, end - int(window * fraction))
    best = None
    for match in _SENTENCE_END.finditer(text, earliest, end):
        best = match.end()
    if best is None or best <= start:
        return end
    return best


class CharWindower:
    """Character windows. No model required."""

    def __init__(self, size: int, overlap: int, *, snap: bool, fraction: float) -> None:
        _validate(size, overlap)
        self.size = size
        self.overlap = overlap
        self.snap = snap
        self.fraction = fraction
        self.identity = "chars"

    def count(self, text: str) -> int:
        return len(text)

    def windows(self, text: str) -> list[tuple[int, int, int]]:
        if not text:
            return []
        out: list[tuple[int, int, int]] = []
        start = 0
        stride = self.size - self.overlap
        while start < len(text):
            end = min(start + self.size, len(text))
            if self.snap:
                end = _snap_end(text, start, end, self.fraction)
            out.append((start, end, end - start))
            if end >= len(text):
                break
            start = max(start + stride, start + 1) if end - start >= stride else end
        return out


class TokenWindower:
    """Windows measured in the embedding model's own tokens."""

    def __init__(
        self,
        model: str,
        size: int,
        overlap: int,
        *,
        snap: bool,
        fraction: float,
    ) -> None:
        _validate(size, overlap)
        self.model = model
        self.size = size
        self.overlap = overlap
        self.snap = snap
        self.fraction = fraction
        self.identity = f"tokens:{model}"
        self._tok: Any = None

    def _tokenizer(self) -> Any:
        if self._tok is None:
            try:
                from transformers import AutoTokenizer
            except ImportError as exc:
                raise TokenizerError(
                    f"transformers is required for chunk.unit=tokens "
                    f"({exc}). Install it, or set chunk.unit=chars to measure in "
                    f"characters instead."
                ) from exc
            tok = AutoTokenizer.from_pretrained(self.model, use_fast=True)
            if not getattr(tok, "is_fast", False):
                # Offset mapping is a fast-tokenizer feature, and without it a
                # chunk's char_span would have to be reconstructed by decoding,
                # which does not round-trip.
                raise TokenizerError(
                    f"{self.model} has no fast tokenizer; character offsets are "
                    f"unavailable and chunk spans could not be sourced."
                )
            self._tok = tok
        return self._tok

    def _encode(self, text: str) -> list[tuple[int, int]]:
        """Character offsets of each content token, special tokens excluded."""
        encoded = self._tokenizer()(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=False,
            verbose=False,
        )
        # Zero-width offsets are special or control tokens; they carry no text
        # and would produce empty spans.
        return [(s, e) for s, e in encoded["offset_mapping"] if e > s]

    def count(self, text: str) -> int:
        return len(self._encode(text))

    def windows(self, text: str) -> list[tuple[int, int, int]]:
        if not text.strip():
            return []
        offsets = self._encode(text)
        if not offsets:
            return []

        out: list[tuple[int, int, int]] = []
        stride = self.size - self.overlap
        i = 0
        while i < len(offsets):
            j = min(i + self.size, len(offsets))
            start = offsets[i][0]
            end = offsets[j - 1][1]

            if self.snap and j < len(offsets):
                snapped = _snap_end(text, start, end, self.fraction)
                if snapped < end:
                    # Keep the token count honest: recount to the snapped end
                    # rather than reporting the pre-snap window length.
                    k = j
                    while k > i + 1 and offsets[k - 1][1] > snapped:
                        k -= 1
                    if k > i:
                        j, end = k, offsets[k - 1][1]

            out.append((start, end, j - i))
            if j >= len(offsets):
                break
            # Advance by stride from the window we actually emitted, so snapping
            # shortens the window without breaking the overlap guarantee.
            i = max(i + 1, j - self.overlap)
        return out


def _validate(size: int, overlap: int) -> None:
    if size <= 0:
        raise TokenizerError(f"chunk.size must be positive, got {size}")
    if overlap < 0:
        raise TokenizerError(f"chunk.overlap must not be negative, got {overlap}")
    if overlap >= size:
        # Otherwise the stride is zero or negative and windowing never advances.
        raise TokenizerError(
            f"chunk.overlap ({overlap}) must be smaller than chunk.size ({size}); "
            f"otherwise each window would start at or before the previous one."
        )


def get_windower(config: Config) -> Windower:
    """Build the windower named by ``chunk.unit``."""
    unit = str(config.get("chunk.unit", "tokens")).strip().lower()
    size = int(config.get("chunk.size"))
    overlap = int(config.get("chunk.overlap"))
    snap = bool(config.get("chunk.snap_to_sentence", True))
    fraction = float(config.get("chunk.snap_search_fraction", 0.2))

    if unit == "chars":
        return CharWindower(size, overlap, snap=snap, fraction=fraction)
    if unit == "tokens":
        model = config.get("embed.providers.local.model")
        return TokenWindower(model, size, overlap, snap=snap, fraction=fraction)
    raise TokenizerError(
        f"unknown chunk.unit {unit!r}; expected 'tokens' or 'chars'."
    )
