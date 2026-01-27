"""Unit tests for embed_facts utilities."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from llm_learn.inference.embed_facts import EmbedFactsResult, embed_missing_facts
from llm_learn.inference.embedder import EmbeddingResult


class TestEmbedFactsResult:
    """Test EmbedFactsResult dataclass."""

    def test_create_result(self):
        """Test creating an EmbedFactsResult."""
        result = EmbedFactsResult(processed=10, failed=1)
        assert result.processed == 10
        assert result.failed == 1


class TestEmbedMissingFacts:
    """Test embed_missing_facts function."""

    @pytest.fixture
    def mock_logger(self):
        """Create a mock Logger."""
        return MagicMock()

    @pytest.fixture
    def mock_embedder(self):
        """Create a mock Embedder."""
        embedder = MagicMock()
        embedder.embed_batch_async = AsyncMock()
        embedder.embed_async = AsyncMock()
        return embedder

    @pytest.fixture
    def mock_facts_client(self):
        """Create a mock FactsClient."""
        client = MagicMock()
        client.list_without_embeddings = MagicMock()
        client.set_embedding = MagicMock()
        return client

    @pytest.fixture
    def sample_facts(self):
        """Create sample facts for testing."""
        facts = []
        for i in range(3):
            fact = MagicMock()
            fact.id = i + 1
            fact.content = f"Fact {i + 1}"
            facts.append(fact)
        return facts

    @pytest.mark.asyncio
    async def test_embed_no_missing_facts(self, mock_logger, mock_embedder, mock_facts_client):
        """Test when no facts need embedding."""
        mock_facts_client.list_without_embeddings.return_value = []

        result = await embed_missing_facts(
            mock_logger, mock_embedder, mock_facts_client, "test-model"
        )

        assert result.processed == 0
        assert result.failed == 0
        mock_embedder.embed_batch_async.assert_not_called()

    @pytest.mark.asyncio
    async def test_embed_single_batch(
        self, mock_logger, mock_embedder, mock_facts_client, sample_facts
    ):
        """Test embedding a single batch of facts."""
        # First call returns facts, second call returns empty (done)
        mock_facts_client.list_without_embeddings.side_effect = [sample_facts, []]

        mock_embedder.embed_batch_async.return_value = [
            EmbeddingResult(embedding=[0.1, 0.2], model="test-model", prompt_tokens=5),
            EmbeddingResult(embedding=[0.3, 0.4], model="test-model", prompt_tokens=5),
            EmbeddingResult(embedding=[0.5, 0.6], model="test-model", prompt_tokens=5),
        ]

        result = await embed_missing_facts(
            mock_logger, mock_embedder, mock_facts_client, "test-model"
        )

        assert result.processed == 3
        assert result.failed == 0
        mock_embedder.embed_batch_async.assert_called_once_with(["Fact 1", "Fact 2", "Fact 3"])
        assert mock_facts_client.set_embedding.call_count == 3

    @pytest.mark.asyncio
    async def test_embed_multiple_batches(
        self, mock_logger, mock_embedder, mock_facts_client, sample_facts
    ):
        """Test embedding multiple batches of facts."""
        # Return facts in two batches
        batch1 = sample_facts[:2]
        batch2 = sample_facts[2:]

        mock_facts_client.list_without_embeddings.side_effect = [batch1, batch2, []]

        mock_embedder.embed_batch_async.side_effect = [
            [
                EmbeddingResult(embedding=[0.1], model="m", prompt_tokens=1),
                EmbeddingResult(embedding=[0.2], model="m", prompt_tokens=1),
            ],
            [
                EmbeddingResult(embedding=[0.3], model="m", prompt_tokens=1),
            ],
        ]

        result = await embed_missing_facts(
            mock_logger, mock_embedder, mock_facts_client, "test-model", batch_size=2
        )

        assert result.processed == 3
        assert result.failed == 0
        assert mock_embedder.embed_batch_async.call_count == 2

    @pytest.mark.asyncio
    async def test_embed_batch_failure_falls_back_to_individual(
        self, mock_logger, mock_embedder, mock_facts_client, sample_facts
    ):
        """Test fallback to individual embedding when batch fails."""
        mock_facts_client.list_without_embeddings.side_effect = [sample_facts, []]

        # Batch fails, individual calls succeed
        mock_embedder.embed_batch_async.side_effect = Exception("Batch API error")
        mock_embedder.embed_async.return_value = EmbeddingResult(
            embedding=[0.1], model="m", prompt_tokens=1
        )

        result = await embed_missing_facts(
            mock_logger, mock_embedder, mock_facts_client, "test-model"
        )

        assert result.processed == 3
        assert result.failed == 0
        assert mock_embedder.embed_async.call_count == 3

    @pytest.mark.asyncio
    async def test_embed_individual_failure_counted(
        self, mock_logger, mock_embedder, mock_facts_client, sample_facts
    ):
        """Test that individual embedding failures are counted."""
        mock_facts_client.list_without_embeddings.side_effect = [sample_facts, []]

        # Batch fails, then 2 succeed and 1 fails
        mock_embedder.embed_batch_async.side_effect = Exception("Batch error")
        mock_embedder.embed_async.side_effect = [
            EmbeddingResult(embedding=[0.1], model="m", prompt_tokens=1),
            Exception("Individual error"),
            EmbeddingResult(embedding=[0.3], model="m", prompt_tokens=1),
        ]

        result = await embed_missing_facts(
            mock_logger, mock_embedder, mock_facts_client, "test-model"
        )

        assert result.processed == 2
        assert result.failed == 1

    @pytest.mark.asyncio
    async def test_embed_set_embedding_failure_counted(
        self, mock_logger, mock_embedder, mock_facts_client, sample_facts
    ):
        """Test that set_embedding failures are counted."""
        mock_facts_client.list_without_embeddings.side_effect = [sample_facts, []]

        mock_embedder.embed_batch_async.return_value = [
            EmbeddingResult(embedding=[0.1], model="m", prompt_tokens=1),
            EmbeddingResult(embedding=[0.2], model="m", prompt_tokens=1),
            EmbeddingResult(embedding=[0.3], model="m", prompt_tokens=1),
        ]

        # Second set_embedding fails
        mock_facts_client.set_embedding.side_effect = [None, Exception("DB error"), None]

        result = await embed_missing_facts(
            mock_logger, mock_embedder, mock_facts_client, "test-model"
        )

        assert result.processed == 2
        assert result.failed == 1

    @pytest.mark.asyncio
    async def test_embed_uses_correct_model_name(
        self, mock_logger, mock_embedder, mock_facts_client, sample_facts
    ):
        """Test that model_name is passed correctly to set_embedding."""
        mock_facts_client.list_without_embeddings.side_effect = [
            sample_facts[:1],
            [],
        ]
        mock_embedder.embed_batch_async.return_value = [
            EmbeddingResult(embedding=[0.1, 0.2, 0.3], model="m", prompt_tokens=1),
        ]

        await embed_missing_facts(
            mock_logger, mock_embedder, mock_facts_client, "custom-model-name"
        )

        mock_facts_client.list_without_embeddings.assert_called_with("custom-model-name", limit=50)
        mock_facts_client.set_embedding.assert_called_once_with(
            1, [0.1, 0.2, 0.3], "custom-model-name"
        )

    @pytest.mark.asyncio
    async def test_embed_custom_batch_size(
        self, mock_logger, mock_embedder, mock_facts_client, sample_facts
    ):
        """Test that custom batch_size is used."""
        mock_facts_client.list_without_embeddings.return_value = []

        await embed_missing_facts(
            mock_logger, mock_embedder, mock_facts_client, "test-model", batch_size=25
        )

        mock_facts_client.list_without_embeddings.assert_called_once_with("test-model", limit=25)

    @pytest.mark.asyncio
    async def test_embed_breaks_on_no_progress(
        self, mock_logger, mock_embedder, mock_facts_client, sample_facts
    ):
        """Test that loop breaks when no progress is made to avoid infinite loop."""
        # Always return the same facts (simulating persistent storage failure)
        mock_facts_client.list_without_embeddings.return_value = sample_facts

        # Batch succeeds but all storage fails
        mock_embedder.embed_batch_async.return_value = [
            EmbeddingResult(embedding=[0.1], model="m", prompt_tokens=1),
            EmbeddingResult(embedding=[0.2], model="m", prompt_tokens=1),
            EmbeddingResult(embedding=[0.3], model="m", prompt_tokens=1),
        ]
        mock_facts_client.set_embedding.side_effect = Exception("DB unavailable")

        result = await embed_missing_facts(
            mock_logger, mock_embedder, mock_facts_client, "test-model"
        )

        # Should have attempted once and stopped
        assert result.processed == 0
        assert result.failed == 3
        assert mock_embedder.embed_batch_async.call_count == 1
        mock_logger.warning.assert_called()
