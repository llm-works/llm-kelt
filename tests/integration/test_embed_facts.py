"""Integration tests for embed_facts utilities."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from llm_learn.inference.embed_facts import embed_missing_facts
from llm_learn.inference.embedder import EmbeddingResult


class TestEmbedMissingFactsIntegration:
    """Integration tests for embed_missing_facts with real database."""

    @pytest.fixture
    def mock_embedder(self):
        """Create a mock Embedder that returns deterministic embeddings."""
        embedder = MagicMock()

        def make_embedding(text):
            """Generate a simple deterministic embedding from text."""
            # Use text length and first char to create a simple embedding
            val = len(text) / 100.0
            return [val, val + 0.1, val + 0.2]

        async def embed_batch_async(texts):
            return [
                EmbeddingResult(
                    embedding=make_embedding(t),
                    model="test-model",
                    prompt_tokens=len(t),
                )
                for t in texts
            ]

        async def embed_async(text):
            return EmbeddingResult(
                embedding=make_embedding(text),
                model="test-model",
                prompt_tokens=len(text),
            )

        embedder.embed_batch_async = AsyncMock(side_effect=embed_batch_async)
        embedder.embed_async = AsyncMock(side_effect=embed_async)
        embedder.discover_async = AsyncMock(return_value="test-model")
        return embedder

    @pytest.fixture
    def sample_facts(self, learn_client, clean_tables):
        """Create a small set of facts for testing."""
        facts = [
            ("User prefers Python", "preferences"),
            ("Timezone is UTC", "settings"),
            ("Output format: markdown", "preferences"),
            ("Experience level: senior", "background"),
            ("Likes concise responses", "preferences"),
        ]
        fact_ids = []
        for content, category in facts:
            fact_id = learn_client.facts.add(content, category=category)
            fact_ids.append(fact_id)
        return fact_ids

    @pytest.mark.asyncio
    async def test_embed_all_facts(self, logger, learn_client, mock_embedder, sample_facts):
        """Test embedding all facts that don't have embeddings."""
        # Verify no embeddings exist
        without = learn_client.facts.list_without_embeddings("test-model")
        assert len(without) == 5

        # Run embedding
        result = await embed_missing_facts(logger, mock_embedder, learn_client.facts)

        assert result.processed == 5
        assert result.failed == 0

        # Verify all facts now have embeddings
        without_after = learn_client.facts.list_without_embeddings("test-model")
        assert len(without_after) == 0

        # Verify embeddings work for similarity search
        search_results = learn_client.facts.search_similar(
            embedding=[0.2, 0.3, 0.4],
            model_name="test-model",
            min_similarity=0.0,
        )
        assert len(search_results) == 5

    @pytest.mark.asyncio
    async def test_embed_skips_already_embedded(
        self, logger, learn_client, mock_embedder, sample_facts
    ):
        """Test that already-embedded facts are skipped."""
        # Embed first two facts manually
        learn_client.facts.set_embedding(sample_facts[0], [0.1, 0.2, 0.3], "test-model")
        learn_client.facts.set_embedding(sample_facts[1], [0.4, 0.5, 0.6], "test-model")

        # Run embedding
        result = await embed_missing_facts(logger, mock_embedder, learn_client.facts)

        # Should only process the 3 without embeddings
        assert result.processed == 3
        assert result.failed == 0

    @pytest.mark.asyncio
    async def test_embed_different_model_embeds_all(
        self, logger, learn_client, mock_embedder, sample_facts
    ):
        """Test that switching models embeds all facts for new model."""
        # Embed all for model-a
        for fact_id in sample_facts:
            learn_client.facts.set_embedding(fact_id, [0.1, 0.2, 0.3], "model-a")

        # Update embedder to discover model-b
        mock_embedder.discover_async = AsyncMock(return_value="model-b")

        # Run embedding - should discover model-b and embed all for it
        result = await embed_missing_facts(logger, mock_embedder, learn_client.facts)

        # Should embed all 5 for the new model
        assert result.processed == 5
        assert result.failed == 0

        # Both models should have embeddings now
        without_a = learn_client.facts.list_without_embeddings("model-a")
        without_b = learn_client.facts.list_without_embeddings("model-b")
        assert len(without_a) == 0
        assert len(without_b) == 0

    @pytest.mark.asyncio
    async def test_embed_respects_batch_size(
        self, logger, learn_client, mock_embedder, sample_facts
    ):
        """Test that batch_size controls how many facts are processed per batch."""
        result = await embed_missing_facts(logger, mock_embedder, learn_client.facts, batch_size=2)

        assert result.processed == 5

        # With 5 facts and batch_size=2, should have called embed_batch 3 times
        # (2 + 2 + 1)
        assert mock_embedder.embed_batch_async.call_count == 3

    @pytest.mark.asyncio
    async def test_embed_inactive_facts_skipped(
        self, logger, learn_client, mock_embedder, clean_tables
    ):
        """Test that inactive facts are not embedded."""
        # Create mix of active and inactive facts
        active_id = learn_client.facts.add("Active fact")
        inactive_id = learn_client.facts.add("Inactive fact")
        learn_client.facts.deactivate(inactive_id)

        result = await embed_missing_facts(logger, mock_embedder, learn_client.facts)

        # Only active fact should be embedded
        assert result.processed == 1

        # Verify via search
        search_results = learn_client.facts.search_similar(
            embedding=[0.1, 0.2, 0.3],
            model_name="test-model",
            min_similarity=0.0,
        )
        assert len(search_results) == 1
        assert search_results[0].fact.id == active_id

    @pytest.mark.asyncio
    async def test_embed_empty_profile(self, logger, learn_client, mock_embedder, clean_tables):
        """Test embedding when profile has no facts."""
        result = await embed_missing_facts(logger, mock_embedder, learn_client.facts)

        assert result.processed == 0
        assert result.failed == 0
        mock_embedder.embed_batch_async.assert_not_called()
