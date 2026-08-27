# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-kelt Authors

"""Integration tests for embed_facts utilities."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from llm_infer.client import EmbeddingResult
from llm_infer.client.backends.embedding import BatchEmbeddingResult

from llm_kelt.inference.embed_facts import embed_missing_facts


class TestEmbedMissingFactsIntegration:
    """Integration tests for embed_missing_facts with real database."""

    @pytest.fixture
    def mock_embedder(self):
        """Create a mock EmbeddingClient that returns deterministic embeddings."""
        embedder = MagicMock()

        def make_embedding(text):
            """Generate a simple deterministic embedding from text."""
            # Use text length and first char to create a simple embedding
            val = len(text) / 100.0
            return [val, val + 0.1, val + 0.2]

        async def embed_batch_async(texts, *, context=None):
            embeddings = [make_embedding(t) for t in texts]
            return BatchEmbeddingResult(
                embeddings=embeddings,
                model="test-model",
                dimensions=3,
                size=len(texts),
                total_prompt_tokens=sum(len(t) for t in texts),
            )

        async def embed_async(text, *, context=None):
            return EmbeddingResult(
                embedding=make_embedding(text),
                model="test-model",
                dimensions=3,
                prompt_tokens=len(text),
            )

        embedder.embed_batch_async = AsyncMock(side_effect=embed_batch_async)
        embedder.embed_async = AsyncMock(side_effect=embed_async)
        embedder.model = "test-model"
        return embedder

    @pytest.fixture
    def sample_facts(self, kelt_client, clean_tables):
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
            fact_id = kelt_client.atomic.assertions.add(content, category=category)
            fact_ids.append(fact_id)
        return fact_ids

    @pytest.mark.asyncio
    async def test_embed_all_facts(self, logger, kelt_client, mock_embedder, sample_facts):
        """Test embedding all facts that don't have embeddings."""
        # Verify no embeddings exist
        without = kelt_client.atomic.embeddings.list_without_embeddings("test-model", 3)
        assert len(without) == 5

        # Run embedding - pass the embeddings adapter
        result = await embed_missing_facts(
            logger, mock_embedder, kelt_client.atomic.embeddings, dimensions=3
        )

        assert result.processed == 5
        assert result.failed == 0

        # Verify all facts now have embeddings
        without_after = kelt_client.atomic.embeddings.list_without_embeddings("test-model", 3)
        assert len(without_after) == 0

        # Verify embeddings work for similarity search
        search_results = kelt_client.atomic.embeddings.search_similar(
            query=[0.2, 0.3, 0.4],
            model="test-model",
            min_similarity=0.0,
        )
        assert len(search_results) == 5

    @pytest.mark.asyncio
    async def test_embed_skips_already_embedded(
        self, logger, kelt_client, mock_embedder, sample_facts
    ):
        """Test that already-embedded facts are skipped."""
        # Embed first two facts manually
        kelt_client.atomic.embeddings.set_embedding(sample_facts[0], [0.1, 0.2, 0.3], "test-model")
        kelt_client.atomic.embeddings.set_embedding(sample_facts[1], [0.4, 0.5, 0.6], "test-model")

        # Run embedding
        result = await embed_missing_facts(
            logger, mock_embedder, kelt_client.atomic.embeddings, dimensions=3
        )

        # Should only process the 3 without embeddings
        assert result.processed == 3
        assert result.failed == 0

    @pytest.mark.asyncio
    async def test_embed_different_model_embeds_all(
        self, logger, kelt_client, mock_embedder, sample_facts
    ):
        """Test that switching models embeds all facts for new model."""
        # Embed all for model-a
        for fact_id in sample_facts:
            kelt_client.atomic.embeddings.set_embedding(fact_id, [0.1, 0.2, 0.3], "model-a")

        # Update embedder model to model-b
        mock_embedder.model = "model-b"

        # Run embedding - should discover model-b and embed all for it
        result = await embed_missing_facts(
            logger, mock_embedder, kelt_client.atomic.embeddings, dimensions=3
        )

        # Should embed all 5 for the new model
        assert result.processed == 5
        assert result.failed == 0

        # Both models should have embeddings now
        without_a = kelt_client.atomic.embeddings.list_without_embeddings("model-a", 3)
        without_b = kelt_client.atomic.embeddings.list_without_embeddings("model-b", 3)
        assert len(without_a) == 0
        assert len(without_b) == 0

    @pytest.mark.asyncio
    async def test_embed_respects_batch_size(
        self, logger, kelt_client, mock_embedder, sample_facts
    ):
        """Test that batch_size controls how many facts are processed per batch."""
        result = await embed_missing_facts(
            logger, mock_embedder, kelt_client.atomic.embeddings, dimensions=3, batch_size=2
        )

        assert result.processed == 5

        # With 5 facts and batch_size=2, should have called embed_batch 3 times
        # (2 + 2 + 1)
        assert mock_embedder.embed_batch_async.call_count == 3

    @pytest.mark.asyncio
    async def test_embed_inactive_facts_skipped(
        self, logger, kelt_client, mock_embedder, clean_tables
    ):
        """Test that inactive facts are not embedded."""
        # Create mix of active and inactive facts
        active_id = kelt_client.atomic.assertions.add("Active fact")
        inactive_id = kelt_client.atomic.assertions.add("Inactive fact")
        kelt_client.atomic.assertions.deactivate(inactive_id)

        result = await embed_missing_facts(
            logger, mock_embedder, kelt_client.atomic.embeddings, dimensions=3
        )

        # Only active fact should be embedded
        assert result.processed == 1

        # Verify via search
        search_results = kelt_client.atomic.embeddings.search_similar(
            query=[0.1, 0.2, 0.3],
            model="test-model",
            min_similarity=0.0,
        )
        assert len(search_results) == 1
        assert search_results[0].entity.id == active_id

    @pytest.mark.asyncio
    async def test_embed_empty_context(self, logger, kelt_client, mock_embedder, clean_tables):
        """Test embedding when context has no facts."""
        result = await embed_missing_facts(
            logger, mock_embedder, kelt_client.atomic.embeddings, dimensions=3
        )

        assert result.processed == 0
        assert result.failed == 0
        mock_embedder.embed_batch_async.assert_not_called()


class TestPublicEmbeddingStore:
    """Test the public embeddings property for custom entity types."""

    def test_embeddings_property_exists(self, kelt_client):
        """Test that embeddings is publicly accessible."""
        from llm_kelt import EmbeddingStoreClient

        store = kelt_client.embeddings
        assert store is not None
        assert isinstance(store, EmbeddingStoreClient)

    def test_store_custom_entity_type(self, kelt_client, clean_tables):
        """Test storing embeddings for a custom entity type."""
        store = kelt_client.embeddings

        # Store embedding for a custom entity type
        store.store(
            entity_type="myapp.query",
            entity_id="q123",
            embedding=[0.1, 0.2, 0.3],
            model="test-model",
        )

        # Verify it exists
        assert store.exists("myapp.query", "q123", "test-model")

        # Retrieve it
        emb = store.get("myapp.query", "q123", "test-model")
        assert emb == [0.1, 0.2, 0.3]

    def test_search_custom_entity_type(self, kelt_client, clean_tables):
        """Test searching embeddings for a custom entity type."""
        store = kelt_client.embeddings

        # Store multiple embeddings
        store.store("myapp.query", "q1", [1.0, 0.0, 0.0], "test-model")
        store.store("myapp.query", "q2", [0.9, 0.1, 0.0], "test-model")
        store.store("myapp.query", "q3", [0.0, 0.0, 1.0], "test-model")

        # Search for similar to q1
        results = store.search(
            query=[1.0, 0.0, 0.0],
            entity_type="myapp.query",
            model="test-model",
            top_k=2,
        )

        # q1 and q2 should be most similar
        entity_ids = [r[0] for r in results]
        assert "q1" in entity_ids
        assert "q2" in entity_ids

    def test_delete_custom_entity_type(self, kelt_client, clean_tables):
        """Test deleting embeddings for a custom entity type."""
        store = kelt_client.embeddings

        # Store and verify
        store.store("myapp.query", "q999", [0.5, 0.5, 0.5], "test-model")
        assert store.exists("myapp.query", "q999", "test-model")

        # Delete
        count = store.delete("myapp.query", "q999")
        assert count == 1

        # Verify gone
        assert not store.exists("myapp.query", "q999", "test-model")

    def test_custom_entity_isolated_from_facts(self, kelt_client, clean_tables):
        """Test that custom entity embeddings don't interfere with fact embeddings."""
        store = kelt_client.embeddings

        # Store custom embedding
        store.store("myapp.query", "1", [0.5, 0.5, 0.5], "test-model")

        # Create a fact with the same embedding
        fact_id = kelt_client.atomic.assertions.add("Test assertion")
        kelt_client.atomic.embeddings.set_embedding(fact_id, [0.5, 0.5, 0.5], "test-model")

        # Search in custom type - should only find custom entity
        custom_results = store.search([0.5, 0.5, 0.5], "myapp.query", "test-model")
        assert len(custom_results) == 1
        assert custom_results[0][0] == "1"

        # Search in facts via atomic adapter - should only find fact
        fact_results = kelt_client.atomic.embeddings.search_similar(
            query=[0.5, 0.5, 0.5], model="test-model", min_similarity=0.0
        )
        assert len(fact_results) == 1
        assert fact_results[0].entity.id == fact_id


class TestCustomTablePrefix:
    """Test custom table prefix for tenant/application isolation."""

    def test_prefix_creates_separate_table(self, database, clean_tables):
        """Test that prefix creates a separate table."""
        from llm_kelt.embedding import Config, Factory, QuantizationFormat

        factory = Factory()

        # Create two clients with different prefixes
        config_a = Config(
            context_key="_test",
            format=QuantizationFormat.F32,
            dimensions=3,
            prefix="tenant_a",
        )
        config_b = Config(
            context_key="_test",
            format=QuantizationFormat.F32,
            dimensions=3,
            prefix="tenant_b",
        )

        client_a = factory.create(database.session, config_a)
        client_b = factory.create(database.session, config_b)

        # Verify table names
        assert config_a.table_name == "embeddings_tenant_a_3_f32"
        assert config_b.table_name == "embeddings_tenant_b_3_f32"

        # Store in tenant_a
        client_a.store("doc", "1", [1.0, 0.0, 0.0], "test-model")

        # tenant_b should not see it
        assert client_a.exists("doc", "1", "test-model")
        assert not client_b.exists("doc", "1", "test-model")

        # Store in tenant_b
        client_b.store("doc", "1", [0.0, 1.0, 0.0], "test-model")

        # Both should exist independently
        emb_a = client_a.get("doc", "1", "test-model")
        emb_b = client_b.get("doc", "1", "test-model")
        assert emb_a == [1.0, 0.0, 0.0]
        assert emb_b == [0.0, 1.0, 0.0]

    def test_no_prefix_uses_default_table(self, database, clean_tables):
        """Test that no prefix uses default table naming."""
        from llm_kelt.embedding import Config, QuantizationFormat

        config = Config(
            context_key="_test",
            format=QuantizationFormat.F16,
            dimensions=384,
        )
        assert config.table_name == "embeddings_384_f16"

        config_with_prefix = Config(
            context_key="_test",
            format=QuantizationFormat.F16,
            dimensions=384,
            prefix="custom",
        )
        assert config_with_prefix.table_name == "embeddings_custom_384_f16"
