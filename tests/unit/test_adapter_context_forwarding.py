"""Verify that context= is threaded to the embedder from the atomic + kg adapters."""

from __future__ import annotations

from unittest.mock import MagicMock

from llm_infer.client import EmbeddingResult

from llm_kelt.memory.atomic.embedding import EmbeddingAdapter as FactEmbeddingAdapter
from llm_kelt.memory.kg.embedding import EntityEmbeddingAdapter


def _stub_embedder(dim: int = 3) -> MagicMock:
    embedder = MagicMock()
    embedder.model = "test-model"
    embedder.embed = MagicMock(
        return_value=EmbeddingResult(
            embedding=[0.1] * dim, model="test-model", dimensions=dim, prompt_tokens=1
        )
    )
    return embedder


def _install_stub_store(adapter) -> MagicMock:
    """Bypass real store creation so we can assert only on the embedder call."""
    store = MagicMock()
    adapter._get_store = MagicMock(return_value=store)
    return store


class TestFactAdapterContextForwarding:
    def test_embed_fact_forwards_context(self):
        adapter = FactEmbeddingAdapter.__new__(FactEmbeddingAdapter)
        adapter._embedder = _stub_embedder()
        adapter._default_dimensions = 3
        _install_stub_store(adapter)

        fact = MagicMock()
        fact.id = 42
        fact.content = "hello"
        ctx = {"tenant": "acme"}

        adapter.embed_fact(fact, context=ctx)

        adapter._embedder.embed.assert_called_once_with("hello", dimensions=3, context=ctx)


class TestEntityAdapterContextForwarding:
    def test_embed_entity_forwards_context(self):
        adapter = EntityEmbeddingAdapter.__new__(EntityEmbeddingAdapter)
        adapter._embedder = _stub_embedder()
        _install_stub_store(adapter)

        entity = MagicMock()
        entity.id = 7
        entity.canonical_name = "Company A"
        entity.description = None
        ctx = {"trace_id": "abc"}

        adapter.embed_entity(entity, context=ctx)

        adapter._embedder.embed.assert_called_once_with("Company A", context=ctx)

    def test_embed_text_forwards_context(self):
        adapter = EntityEmbeddingAdapter.__new__(EntityEmbeddingAdapter)
        adapter._embedder = _stub_embedder()
        ctx = {"trace_id": "xyz"}

        result = adapter.embed_text("query text", context=ctx)

        assert result == [0.1, 0.1, 0.1]
        adapter._embedder.embed.assert_called_once_with("query text", context=ctx)
