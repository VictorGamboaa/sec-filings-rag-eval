"""Local embedder over sentence-transformers. The default implementation.

Chosen as the default because it has no per-call cost, which matters for a
harness whose purpose is repeated index builds at varying scale: an embedding
provider that bills per token turns every scaling run into a spending decision
and makes cost a confound in the throughput measurement.

Implements the contract in ``tools.embedders.base``. Reports zero requests and
zero tokens to the run log -- see contract point 5; that zero is deliberate and
load-bearing for later provider comparison.

Heavy dependencies (numpy, sentence-transformers, torch) are imported lazily so
that listing and inspecting embedders works before ``uv sync`` has run.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tools.embedders.base import EmbedderError, register_embedder

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np

    from tools.config import Config
    from tools.runlog import StageRecorder

__all__ = ["LocalEmbedder"]

#: Vector widths for models we ship a default for. Contract point 1 requires
#: ``dim`` to be known before the first call; for anything not listed here the
#: model is loaded eagerly at construction to read its true width, rather than
#: guessing one.
def _diagnose_import(module: str, exc: ImportError) -> str:
    """Explain an import failure accurately.

    An ImportError does not mean "not installed". A package can be present and
    still fail to import -- most often because a native extension DLL cannot be
    loaded. Reporting every ImportError as a missing dependency sends whoever
    hits it to reinstall a package that was never the problem, which is a
    genuinely expensive wrong turn.
    """
    import importlib.util

    installed = importlib.util.find_spec(module) is not None
    detail = str(exc)

    if not installed:
        return f"{module} is not installed. Run: uv sync"

    if "DLL load failed" in detail or "Application Control" in detail:
        return (
            f"{module} is installed but cannot load a native extension:\n"
            f"    {detail}\n"
            f"  This is a host security policy blocking an unsigned .pyd/.dll, "
            f"not a packaging fault.\n"
            f"  Reinstalling will not help. Check Windows Smart App Control / "
            f"WDAC policy,\n"
            f"  or run the harness where that policy does not apply (WSL, a "
            f"container, or CI)."
        )

    return f"{module} is installed but failed to import: {detail}"


_KNOWN_DIMS: dict[str, int] = {
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-large-en-v1.5": 1024,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "sentence-transformers/all-mpnet-base-v2": 768,
}


class LocalEmbedder:
    """sentence-transformers backend running on local hardware."""

    name = "local"

    def __init__(
        self,
        model: str = "BAAI/bge-small-en-v1.5",
        *,
        device: str = "auto",
        normalize: bool = True,
        max_batch: int = 64,
        query_prefix: str = "",
    ) -> None:
        self.model_name = model
        self.device = None if device == "auto" else device
        self.normalized = bool(normalize)
        self.max_batch = int(max_batch)
        self.query_prefix = query_prefix
        self._model: Any = None

        known = _KNOWN_DIMS.get(model)
        if known is not None:
            self.dim = known
        else:
            # Unknown model: load now and read the real width. Never guess --
            # a wrong dim would mis-allocate the index and fail obscurely later.
            self.dim = int(self._ensure_model().get_sentence_embedding_dimension())

    def _ensure_model(self) -> Any:
        """Load the model on first use.

        Deferred so that constructing an embedder to read its metadata does not
        pay the multi-second model load.
        """
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise EmbedderError(_diagnose_import("sentence_transformers", exc)) from exc
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def _encode(self, texts: list[str], stage: StageRecorder | None) -> np.ndarray:
        import numpy as np

        model = self._ensure_model()
        vectors = model.encode(
            texts,
            batch_size=self.max_batch,
            normalize_embeddings=self.normalized,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        out = np.asarray(vectors, dtype="float32")
        if out.ndim != 2 or out.shape[1] != self.dim:
            # Contract point 1: the declared width is authoritative. If the model
            # disagrees, stop -- do not pad, truncate, or re-declare.
            raise EmbedderError(
                f"{self.model_name} produced shape {out.shape}, expected "
                f"(*, {self.dim}). Index would be built against a wrong width."
            )
        if stage is not None:
            stage.count(len(texts))
            stage.bytes_out(int(out.nbytes))
            # No request()/tokens() calls: local inference makes no network
            # requests and bills no tokens. Contract point 5.
        return out

    def embed_documents(
        self, texts: list[str], stage: StageRecorder | None = None
    ) -> np.ndarray:
        """Embed corpus chunks. Documents take no prefix for bge models."""
        import numpy as np

        if not texts:
            return np.empty((0, self.dim), dtype="float32")
        return self._encode(list(texts), stage)

    def embed_query(self, text: str, stage: StageRecorder | None = None) -> np.ndarray:
        """Embed one query, applying the model's query prefix if configured.

        bge models are trained with an asymmetric query instruction; omitting it
        measurably degrades retrieval. This is exactly the asymmetry contract
        point 2 exists to preserve.
        """
        return self._encode([self.query_prefix + text], stage)[0]

    def describe(self) -> dict[str, Any]:
        return {
            "provider": self.name,
            "model": self.model_name,
            "dim": self.dim,
            "normalized": self.normalized,
            "max_batch": self.max_batch,
            "device": self.device or "auto",
            "query_prefix": self.query_prefix or None,
        }

    def __repr__(self) -> str:
        return f"LocalEmbedder(model={self.model_name!r}, dim={self.dim})"


@register_embedder("local")
def _build_local(config: Config) -> LocalEmbedder:
    """Factory: construct a LocalEmbedder from config.

    Batch size comes from ``embed.batch_size`` (Rule 3) rather than the
    provider block, so a scaling sweep can vary it with --set without caring
    which provider is selected.
    """
    settings = config.get("embed.providers.local", {}) or {}
    return LocalEmbedder(
        model=settings.get("model", "BAAI/bge-small-en-v1.5"),
        device=settings.get("device", "auto"),
        normalize=settings.get("normalize", True),
        max_batch=config.get("embed.batch_size", 64),
        query_prefix=settings.get("query_prefix", "") or "",
    )
