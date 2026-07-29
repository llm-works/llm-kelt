"""Verify context= is threaded from atomic write methods to the embedding adapter."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


def _install_mock_adapter(kelt_client) -> MagicMock:
    """Replace the embedding adapter on every FactClient with a MagicMock.

    kelt_client's fixture doesn't wire an embedder, so _embedding_adapter is
    None and _auto_embed_fact short-circuits. Swap in a MagicMock so we can
    observe the embed_fact call the write method triggers.
    """
    adapter = MagicMock()
    atomic = kelt_client.atomic
    for client in (
        atomic.assertions,
        atomic.solutions,
        atomic.predictions,
        atomic.directives,
    ):
        client._embedding_adapter = adapter
    return adapter


@pytest.fixture
def mock_adapter(kelt_client, clean_tables):
    return _install_mock_adapter(kelt_client)


class TestWriteMethodContextForwarding:
    def test_assertions_add_forwards_context(self, kelt_client, mock_adapter):
        ctx = {"tenant": "acme", "trace_id": "a1"}
        kelt_client.atomic.assertions.add("hello", context=ctx)

        mock_adapter.embed_fact.assert_called_once()
        assert mock_adapter.embed_fact.call_args.kwargs["context"] == ctx

    def test_solutions_record_forwards_context(self, kelt_client, mock_adapter):
        ctx = {"tenant": "acme", "trace_id": "s1"}
        kelt_client.atomic.solutions.record(
            agent_name="reviewer",
            problem="review PR",
            problem_context={"messages": []},
            answer={"verdict": "approved"},
            tokens_used=100,
            latency_ms=200,
            context=ctx,
        )

        mock_adapter.embed_fact.assert_called_once()
        assert mock_adapter.embed_fact.call_args.kwargs["context"] == ctx

    def test_predictions_record_forwards_context(self, kelt_client, mock_adapter):
        ctx = {"tenant": "acme", "trace_id": "p1"}
        kelt_client.atomic.predictions.record(
            hypothesis="BTC over 100k by year end",
            confidence=0.6,
            context=ctx,
        )

        mock_adapter.embed_fact.assert_called_once()
        assert mock_adapter.embed_fact.call_args.kwargs["context"] == ctx

    def test_directives_record_forwards_context(self, kelt_client, mock_adapter):
        ctx = {"tenant": "acme", "trace_id": "d1"}
        kelt_client.atomic.directives.record(
            text="always use type hints",
            context=ctx,
        )

        mock_adapter.embed_fact.assert_called_once()
        assert mock_adapter.embed_fact.call_args.kwargs["context"] == ctx

    def test_default_context_is_none(self, kelt_client, mock_adapter):
        """Absent context= at the call site, the adapter sees context=None."""
        kelt_client.atomic.assertions.add("plain add")
        assert mock_adapter.embed_fact.call_args.kwargs["context"] is None
