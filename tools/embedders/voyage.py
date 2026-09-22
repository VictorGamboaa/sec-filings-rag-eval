"""Voyage AI embedder. A hosted, finance-domain provider behind the same seam.

THE bge PREFIX DOES NOT TRANSFER, AND THIS IS WHERE THAT IS ENFORCED
====================================================================

bge encodes the query/document asymmetry as a literal string prepended to
queries: "Represent this sentence for searching relevant passages: ". Voyage
encodes the same asymmetry as an ``input_type`` parameter -- "query" or
"document" -- and prepends its own prompt server-side.

Sending the bge prefix to Voyage would embed the instruction as content. The
call would succeed, the vectors would be well-formed, and the comparison
between providers would be quietly measuring the prefix rather than the model.
So this implementation:

  * passes input_type="document" for corpus text and "query" for queries
  * NEVER prepends a prefix
  * refuses, at the point of sending, any text carrying a known prefix

That last check is deliberate belt-and-braces. The seam is what makes provider
behaviour swappable; a caller that hard-coded bge's convention would otherwise
silently corrupt this provider, and the failure would look like a bad model
rather than a bad call site.

COST AND INSTRUMENTATION
========================

Contract point 5: a network-backed embedder MUST report requests and tokens.
Every API call increments ``stage.request()`` and the provider's own reported
token usage goes to ``stage.tokens()``. The local providers report zero for
both, and that asymmetry is what makes a cost comparison between them legible.
"""

from __future__ import annotations

import os
import random
import time
from typing import TYPE_CHECKING, Any

from tools.embedders.base import EmbedderError, register_embedder

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np

    from tools.config import Config
    from tools.runlog import StageRecorder

__all__ = ["VoyageEmbedder", "FORBIDDEN_PREFIXES"]

#: Prefixes that belong to other providers' conventions. Any of these arriving
#: at this provider means a caller applied the wrong convention.
FORBIDDEN_PREFIXES: tuple[str, ...] = (
    "Represent this sentence for searching relevant passages:",
    "Represent this question for searching relevant passages:",
    "query:",
    "passage:",
)

#: Published per-model figures. Verified against Voyage's own documentation
#: rather than assumed; anything unlisted is rejected rather than guessed.
_KNOWN: dict[str, dict[str, int]] = {
    # docs.voyageai.com/docs/embeddings: 32,000 context, 1024 dimensions.
    "voyage-finance-2": {"dim": 1024, "max_seq_tokens": 32000},
}


class VoyageEmbedder:
    """Voyage AI embeddings over the shared Embedder contract."""

    name = "voyage"

    def __init__(
        self,
        model: str = "voyage-finance-2",
        *,
        api_key: str | None = None,
        normalize: bool = True,
        max_batch: int = 128,
        max_batch_tokens: int = 100_000,
        max_retries: int = 6,
        backoff_initial_sec: float = 2.0,
        backoff_max_sec: float = 60.0,
        dim: int | None = None,
        max_seq_tokens: int | None = None,
    ) -> None:
        if not api_key:
            raise EmbedderError(
                "VOYAGE_API_KEY is not set. Put it in .env (which is gitignored) "
                "and nowhere else. Refusing to build a partial or mocked index."
            )
        self._api_key = api_key  # never logged, never in describe()
        self.model_name = model
        self.normalized = bool(normalize)
        self.max_batch = int(max_batch)
        self.max_batch_tokens = int(max_batch_tokens)
        self.max_retries = int(max_retries)
        self.backoff_initial = float(backoff_initial_sec)
        self.backoff_max = float(backoff_max_sec)
        self.total_tokens = 0
        self.total_requests = 0
        self._client: Any = None

        known = _KNOWN.get(model, {})
        self.dim = int(dim or known.get("dim") or 0)
        self.max_seq_tokens = int(max_seq_tokens or known.get("max_seq_tokens") or 0)
        if not self.dim or not self.max_seq_tokens:
            raise EmbedderError(
                f"no published dim/context recorded for {model!r}. Add it to "
                f"_KNOWN from the provider's documentation rather than guessing "
                f"-- a wrong width mis-allocates the index."
            )

    # -- client ------------------------------------------------------------

    def _ensure_client(self) -> Any:
        if self._client is None:
            try:
                import voyageai
            except ImportError as exc:
                raise EmbedderError(
                    f"voyageai is not installed ({exc}). Run: uv sync"
                ) from exc
            self._client = voyageai.Client(api_key=self._api_key)
        return self._client

    def count_tokens(self, texts: list[str]) -> int:
        """Token count under Voyage's own tokenizer, not another model's."""
        return int(self._ensure_client().count_tokens(texts, model=self.model_name))

    # -- the guard ---------------------------------------------------------

    @staticmethod
    def assert_no_foreign_prefix(texts: list[str]) -> None:
        """Refuse text carrying another provider's query-prefix convention.

        Raised rather than stripped: stripping would hide a call site that is
        applying the wrong convention, and the next provider added would meet
        the same bug.
        """
        for text in texts:
            head = str(text).lstrip()
            for prefix in FORBIDDEN_PREFIXES:
                if head.startswith(prefix):
                    raise EmbedderError(
                        f"text sent to {VoyageEmbedder.name} begins with the "
                        f"prefix {prefix!r}, which is another provider's "
                        f"convention. Voyage expresses query/document asymmetry "
                        f"through input_type, not a prepended string. Embedding "
                        f"this would measure the prefix, not the model."
                    )

    # -- encoding ----------------------------------------------------------

    def _encode(
        self, texts: list[str], input_type: str, stage: StageRecorder | None
    ) -> np.ndarray:
        import numpy as np

        self.assert_no_foreign_prefix(texts)
        client = self._ensure_client()
        out: list[list[float]] = []

        for batch in self._batches(texts):
            vectors = self._embed_batch(client, batch, input_type, stage)
            out.extend(vectors)

        arr = np.asarray(out, dtype="float32")
        if arr.ndim != 2 or arr.shape[1] != self.dim:
            raise EmbedderError(
                f"{self.model_name} produced shape {arr.shape}, expected "
                f"(*, {self.dim}). Index would be built against a wrong width."
            )
        if self.normalized and len(arr):
            norms = np.linalg.norm(arr, axis=1)
            if not np.allclose(norms, 1.0, atol=1e-3):
                # Contract point 3: an implementation that claims normalization
                # and does not deliver it produces a silently wrong ranking.
                raise EmbedderError(
                    f"{self.model_name} returned vectors with norms in "
                    f"[{norms.min():.4f}, {norms.max():.4f}] while normalized=True "
                    f"was declared. Inner product would not be cosine."
                )
        if stage is not None:
            stage.count(len(texts))
            stage.bytes_out(int(arr.nbytes))
        return arr

    def _batches(self, texts: list[str]) -> list[list[str]]:
        """Split by both the count and token ceilings the API imposes."""
        batches: list[list[str]] = []
        current: list[str] = []
        current_tokens = 0
        for text in texts:
            # Cheap upper bound; the exact count is only needed to stay under
            # the per-request ceiling, and over-estimating merely sends smaller
            # batches.
            approx = max(1, len(text) // 3)
            if current and (
                len(current) >= self.max_batch
                or current_tokens + approx > self.max_batch_tokens
            ):
                batches.append(current)
                current, current_tokens = [], 0
            current.append(text)
            current_tokens += approx
        if current:
            batches.append(current)
        return batches

    def _embed_batch(
        self, client: Any, batch: list[str], input_type: str,
        stage: StageRecorder | None,
    ) -> list[list[float]]:
        last = "no attempt made"
        for attempt in range(self.max_retries + 1):
            try:
                self.total_requests += 1
                if stage is not None:
                    stage.request()
                result = client.embed(batch, model=self.model_name,
                                      input_type=input_type)
            except Exception as exc:  # provider SDK raises its own types
                last = f"{type(exc).__name__}: {exc}"
                if attempt >= self.max_retries:
                    break
                delay = min(self.backoff_initial * (2 ** attempt), self.backoff_max)
                time.sleep(delay * (0.5 + random.random() / 2.0))
                continue
            used = int(getattr(result, "total_tokens", 0) or 0)
            self.total_tokens += used
            if stage is not None and used:
                stage.tokens(prompt=used)
            return result.embeddings
        raise EmbedderError(
            f"voyage embed failed after {self.max_retries + 1} attempt(s): {last}. "
            f"Stopping rather than writing a partial index -- an index that "
            f"reports a full chunk count while missing vectors is the worst "
            f"available outcome."
        )

    # -- contract ----------------------------------------------------------

    def embed_documents(
        self, texts: list[str], stage: StageRecorder | None = None
    ) -> np.ndarray:
        import numpy as np

        if not texts:
            return np.empty((0, self.dim), dtype="float32")
        return self._encode(list(texts), "document", stage)

    def embed_query(self, text: str, stage: StageRecorder | None = None) -> np.ndarray:
        """No prefix. The asymmetry is input_type='query', server-side."""
        return self._encode([text], "query", stage)[0]

    def describe(self) -> dict[str, Any]:
        """Identifying metadata. The API key is never included."""
        return {
            "provider": self.name,
            "model": self.model_name,
            "dim": self.dim,
            "normalized": self.normalized,
            "max_batch": self.max_batch,
            "max_seq_tokens": self.max_seq_tokens,
            "query_prefix": None,
            "input_type_query": "query",
            "input_type_document": "document",
        }

    def __repr__(self) -> str:
        return f"VoyageEmbedder(model={self.model_name!r}, dim={self.dim})"


@register_embedder("voyage")
def _build_voyage(config: Config) -> VoyageEmbedder:
    settings = config.get("embed.providers.voyage", {}) or {}
    key_env = settings.get("api_key_env", "VOYAGE_API_KEY")
    return VoyageEmbedder(
        model=settings.get("model", "voyage-finance-2"),
        api_key=os.environ.get(key_env),
        normalize=settings.get("normalize", True),
        max_batch=config.get("embed.batch_size", 128),
        max_batch_tokens=settings.get("max_batch_tokens", 100_000),
        max_retries=settings.get("max_retries", 6),
        backoff_initial_sec=settings.get("backoff_initial_sec", 2.0),
        backoff_max_sec=settings.get("backoff_max_sec", 60.0),
    )
