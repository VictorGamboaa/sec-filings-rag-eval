"""The embedder seam: interface, registry, and contract.

The embed stage must not know which embedder it is using. It resolves one from
``embed.provider`` in config and calls this interface. Adding a hosted provider
(Voyage, OpenAI, Cohere) means writing one class here and registering it -- no
change to the stage, the index, or the evaluation.

Only the local implementation exists today. This module is the seam it and its
successors share.

THE CONTRACT
============

An implementation must honour five things. Each exists because a hosted provider
would otherwise diverge from the local one in a way that silently corrupts a
comparison between them.

1. ``dim``, ``normalized`` and ``max_seq_tokens`` are known *before* any text is
   embedded.
   The FAISS index must be allocated with the right width and the right metric
   at construction. An embedder that only learns its own dimension after the
   first call cannot be indexed against without a wasted probe call.

   ``max_seq_tokens`` is on this list for a sharper reason than the other two.
   Exceeding it is not an error in any embedding library -- the input is
   silently truncated and a perfectly well-formed vector comes back for the
   first N tokens. BAAI/bge-small-en-v1.5 accepts 512; a 1000-token chunk sent
   to it yields a vector representing roughly half the text, with nothing in the
   return value indicating that the rest was dropped. Every downstream
   measurement then reports a full corpus. Declaring the limit up front is what
   lets the chunker size its output against it and the index stage refuse an
   over-length chunk instead of truncating it.

2. Queries and documents are embedded by separate methods.
   The local bge model treats them near-identically apart from a query prefix,
   but that is not general: hosted providers expose distinct ``input_type``
   parameters, charge differently, and in some cases use genuinely different
   towers. Splitting the methods now costs nothing; retrofitting the split later
   would touch every call site in the stage, the retriever, and the evaluator.

3. Normalization is the implementation's responsibility, and it must declare
   what it did via ``normalized``.
   The index layer uses this to decide whether inner product equals cosine.
   An implementation that returns unnormalized vectors while claiming otherwise
   produces a silently wrong ranking -- no error, just degraded results, which
   is the worst possible failure mode for an evaluation harness.

4. Batching belongs to the implementation.
   Callers hand over the whole list. The implementation splits it to respect its
   own ``max_batch``, whether that ceiling comes from VRAM or from an API limit.
   This keeps the stage's loop identical across providers.

5. Instrumentation is part of the interface, not an afterthought.
   ``embed_documents`` takes an optional ``StageRecorder`` and every
   implementation reports through it. Network-backed implementations MUST call
   ``stage.request()`` per HTTP call and ``stage.tokens()`` where the provider
   reports usage. The local implementation reports neither, and that asymmetry
   is the point: a run log showing zero requests against one showing thousands
   is what makes the local-vs-hosted cost and throughput comparison legible.

Rule 4 applies with force here. If a provider fails on a batch, the
implementation logs the error and propagates or returns fewer vectors -- it must
never substitute a zero vector, a mean vector, or a retry result from different
text. A fabricated embedding has no source and no audit standing, and it would
pollute every retrieval metric computed downstream.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Protocol, runtime_checkable

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np

    from tools.config import Config
    from tools.runlog import StageRecorder

__all__ = [
    "Embedder",
    "EmbedderError",
    "register_embedder",
    "get_embedder",
    "available_embedders",
]


class EmbedderError(RuntimeError):
    """An embedder could not be constructed or failed while embedding."""


@runtime_checkable
class Embedder(Protocol):
    """Interface every embedding backend implements.

    See the module docstring for the full contract.
    """

    #: Registry key, matching ``embed.provider`` in config.
    name: str
    #: Vector width. Known before any call; never inferred from output.
    dim: int
    #: Largest list the implementation will send in one underlying call.
    max_batch: int
    #: True if returned vectors are L2-normalized, making inner product == cosine.
    normalized: bool
    #: Longest input in tokens, including any special tokens the model adds.
    #: Text beyond this is truncated by the backend WITHOUT error, so callers
    #: must validate against it rather than discover it. See contract point 1.
    max_seq_tokens: int

    def embed_documents(
        self, texts: list[str], stage: StageRecorder | None = None
    ) -> np.ndarray:
        """Embed corpus chunks. Returns float32 of shape ``(len(texts), dim)``.

        Row order matches input order. The implementation handles its own
        batching against ``max_batch`` and reports counters through ``stage``.
        """
        ...

    def embed_query(self, text: str, stage: StageRecorder | None = None) -> np.ndarray:
        """Embed one search query. Returns float32 of shape ``(dim,)``.

        Separate from ``embed_documents`` by contract, even where an
        implementation treats the two identically.
        """
        ...

    def describe(self) -> dict[str, Any]:
        """Identifying metadata for the run log and index sidecar.

        Must be enough to tell whether two indexes are comparable: at minimum
        the provider name, model identifier, dimension and normalization.
        """
        ...


#: provider name -> factory taking (Config) and returning an Embedder.
_REGISTRY: dict[str, Callable[[Config], Embedder]] = {}


def register_embedder(
    name: str,
) -> Callable[[Callable[[Config], Embedder]], Callable[[Config], Embedder]]:
    """Decorator registering an embedder factory under ``name``."""

    def decorator(factory: Callable[[Config], Embedder]) -> Callable[[Config], Embedder]:
        if name in _REGISTRY:
            raise EmbedderError(f"embedder {name!r} is already registered")
        _REGISTRY[name] = factory
        return factory

    return decorator


def available_embedders() -> list[str]:
    """Names of all registered embedders."""
    _load_builtins()
    return sorted(_REGISTRY)


def get_embedder(config: Config, stage: StageRecorder | None = None) -> Embedder:
    """Construct the embedder named by ``embed.provider``.

    Raises EmbedderError listing the valid names if the provider is unknown --
    a misconfigured run should fail immediately and legibly, not fall back to a
    default, which would make the run log lie about what produced the vectors.
    """
    _load_builtins()
    provider = config.get("embed.provider")
    factory = _REGISTRY.get(provider)
    if factory is None:
        known = ", ".join(sorted(_REGISTRY)) or "(none registered)"
        raise EmbedderError(
            f"unknown embedder {provider!r}. Registered: {known}. "
            f"Set embed.provider in config, or implement the interface in "
            f"tools/embedders/base.py and register it."
        )
    embedder = factory(config)
    if stage is not None:
        stage.note(**{f"embedder_{k}": v for k, v in embedder.describe().items()})
    return embedder


def _load_builtins() -> None:
    """Import bundled implementations so their decorators run.

    Import errors are swallowed on purpose: a missing optional dependency should
    surface as "provider unavailable" when that provider is actually requested,
    not as a crash when merely listing what exists.
    """
    for module in ("local", "voyage"):
        try:
            __import__(f"tools.embedders.{module}")
        except ImportError:
            # A provider whose optional dependency is absent stays unavailable
            # until it is actually requested, rather than crashing a listing.
            pass
