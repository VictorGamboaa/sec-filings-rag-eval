"""Tests for the Voyage provider, with no network calls.

The central one is the prefix guard. bge's query convention is a literal string
prepended to the text; Voyage's is an input_type parameter. Sending bge's prefix
to Voyage would succeed, return well-formed vectors, and make a provider
comparison measure the prefix instead of the model -- so it must raise, and that
has to be a test rather than a comment.
"""

from __future__ import annotations

import numpy as np
import pytest

from tools.embedders import EmbedderError, available_embedders
from tools.embedders.voyage import FORBIDDEN_PREFIXES, VoyageEmbedder

BGE_PREFIX = "Represent this sentence for searching relevant passages: "


def make(**kw):
    return VoyageEmbedder(api_key="test-key-not-used", **kw)


class FakeResult:
    def __init__(self, n, dim, tokens=7):
        rng = np.random.default_rng(0)
        v = rng.standard_normal((n, dim))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        self.embeddings = v.tolist()
        self.total_tokens = tokens


class FakeClient:
    """Records exactly what was sent, so assertions are about the wire."""

    def __init__(self, dim=1024):
        self.dim = dim
        self.calls = []

    def embed(self, texts, model=None, input_type=None):
        self.calls.append({"texts": list(texts), "model": model,
                           "input_type": input_type})
        return FakeResult(len(texts), self.dim)

    def count_tokens(self, texts, model=None):
        return sum(len(t.split()) for t in texts)


# --------------------------------------------------------------------------
# The prefix guard
# --------------------------------------------------------------------------


@pytest.mark.parametrize("prefix", FORBIDDEN_PREFIXES)
def test_foreign_prefix_is_refused(prefix):
    with pytest.raises(EmbedderError, match="another provider's convention"):
        VoyageEmbedder.assert_no_foreign_prefix([f"{prefix} what did Honeywell say?"])


def test_bge_prefix_never_reaches_the_api(monkeypatch):
    """The end-to-end version: a prefixed query must not be sent."""
    e = make()
    client = FakeClient()
    monkeypatch.setattr(e, "_ensure_client", lambda: client)
    with pytest.raises(EmbedderError):
        e.embed_query(BGE_PREFIX + "How much did Honeywell pay?")
    assert client.calls == [], "nothing may reach the API once the guard trips"


def test_leading_whitespace_does_not_evade_the_guard():
    with pytest.raises(EmbedderError):
        VoyageEmbedder.assert_no_foreign_prefix(["   " + BGE_PREFIX + "x"])


def test_ordinary_text_passes_the_guard():
    VoyageEmbedder.assert_no_foreign_prefix([
        "How much did Honeywell pay for Access Solutions?",
        "On June 3, 2024, the Company acquired 100 % of ...",
        "Representing the interests of shareholders",  # near-miss, must pass
    ])


def test_no_text_sent_to_the_api_carries_any_forbidden_prefix(monkeypatch):
    """Sweep every call made during a mixed document+query workload."""
    e = make()
    client = FakeClient()
    monkeypatch.setattr(e, "_ensure_client", lambda: client)
    e.embed_documents(["chunk one", "chunk two"])
    e.embed_query("what changed between 2024 and 2026?")
    sent = [t for call in client.calls for t in call["texts"]]
    assert sent, "the workload must actually have sent something"
    for text in sent:
        for prefix in FORBIDDEN_PREFIXES:
            assert not text.lstrip().startswith(prefix)


# --------------------------------------------------------------------------
# input_type is the asymmetry, in both directions
# --------------------------------------------------------------------------


def test_documents_use_input_type_document(monkeypatch):
    e = make()
    client = FakeClient()
    monkeypatch.setattr(e, "_ensure_client", lambda: client)
    e.embed_documents(["a", "b"])
    assert client.calls[0]["input_type"] == "document"


def test_queries_use_input_type_query(monkeypatch):
    e = make()
    client = FakeClient()
    monkeypatch.setattr(e, "_ensure_client", lambda: client)
    e.embed_query("a question")
    assert client.calls[0]["input_type"] == "query"


def test_query_text_is_sent_unmodified(monkeypatch):
    """No prefix is added -- the asymmetry is the parameter, not the string."""
    e = make()
    client = FakeClient()
    monkeypatch.setattr(e, "_ensure_client", lambda: client)
    q = "What did Honeywell's CEO say about second quarter 2025 results?"
    e.embed_query(q)
    assert client.calls[0]["texts"] == [q]


# --------------------------------------------------------------------------
# Contract
# --------------------------------------------------------------------------


def test_registered_alongside_the_local_providers():
    names = available_embedders()
    assert "voyage" in names
    assert "local" in names and "local_base" in names


def test_missing_api_key_refuses_rather_than_mocking():
    with pytest.raises(EmbedderError, match="VOYAGE_API_KEY"):
        VoyageEmbedder(api_key=None)


def test_describe_never_leaks_the_key():
    e = make()
    blob = repr(e.describe()) + repr(e)
    assert "test-key-not-used" not in blob
    assert e.describe()["query_prefix"] is None
    assert e.describe()["dim"] == 1024


def test_unknown_model_is_refused_rather_than_guessed():
    with pytest.raises(EmbedderError, match="no published dim/context"):
        VoyageEmbedder(api_key="k", model="voyage-does-not-exist")


def test_unnormalized_response_is_refused(monkeypatch):
    """Declaring normalized and not delivering it is a silent ranking bug."""
    e = make()

    class Unnormalized(FakeClient):
        def embed(self, texts, model=None, input_type=None):
            r = FakeResult(len(texts), self.dim)
            r.embeddings = [[v * 3.0 for v in row] for row in r.embeddings]
            return r

    monkeypatch.setattr(e, "_ensure_client", lambda: Unnormalized())
    with pytest.raises(EmbedderError, match="normalized=True"):
        e.embed_documents(["x"])


def test_wrong_width_is_refused(monkeypatch):
    e = make()
    monkeypatch.setattr(e, "_ensure_client", lambda: FakeClient(dim=768))
    with pytest.raises(EmbedderError, match="expected"):
        e.embed_documents(["x"])


def test_tokens_and_requests_are_reported(monkeypatch):
    """Contract point 5: a hosted provider must report both."""
    e = make()
    monkeypatch.setattr(e, "_ensure_client", lambda: FakeClient())

    class Stage:
        def __init__(self):
            self.requests = 0
            self.prompt = 0
            self.items = 0

        def request(self, n=1):
            self.requests += n

        def tokens(self, prompt=0, completion=0):
            self.prompt += prompt

        def count(self, n=1):
            self.items += n

        def bytes_out(self, n):
            pass

    stage = Stage()
    e.embed_documents(["a", "b", "c"], stage)
    assert stage.requests >= 1
    assert stage.prompt > 0
    assert stage.items == 3


def test_persistent_failure_raises_rather_than_returning_short(monkeypatch):
    """A partial index reporting a full count is the worst outcome."""
    e = make(max_retries=1, backoff_initial_sec=0.0, backoff_max_sec=0.0)

    class Broken(FakeClient):
        def embed(self, texts, model=None, input_type=None):
            raise RuntimeError("503 upstream")

    monkeypatch.setattr(e, "_ensure_client", lambda: Broken())
    with pytest.raises(EmbedderError, match="partial index"):
        e.embed_documents(["x"])


def test_batching_respects_the_count_ceiling(monkeypatch):
    e = make(max_batch=2)
    client = FakeClient()
    monkeypatch.setattr(e, "_ensure_client", lambda: client)
    e.embed_documents([f"text {i}" for i in range(5)])
    assert [len(c["texts"]) for c in client.calls] == [2, 2, 1]
