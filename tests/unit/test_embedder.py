"""Unit tests for Embedder client."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from llm_learn.inference.embedder import Embedder, EmbeddingResult


class TestEmbeddingResult:
    """Test EmbeddingResult dataclass."""

    def test_create_result(self):
        """Test creating an EmbeddingResult."""
        result = EmbeddingResult(
            embedding=[0.1, 0.2, 0.3],
            model="test-model",
            prompt_tokens=10,
        )
        assert result.embedding == [0.1, 0.2, 0.3]
        assert result.model == "test-model"
        assert result.prompt_tokens == 10


class TestEmbedder:
    """Test Embedder client."""

    @pytest.fixture
    def mock_response(self):
        """Create a mock HTTP response."""
        response = MagicMock()
        response.json.return_value = {
            "data": [{"embedding": [0.1, 0.2, 0.3], "index": 0}],
            "model": "all-MiniLM-L6-v2",
            "usage": {"prompt_tokens": 5},
        }
        response.raise_for_status = MagicMock()
        return response

    @pytest.fixture
    def mock_batch_response(self):
        """Create a mock HTTP response for batch embeddings."""
        response = MagicMock()
        response.json.return_value = {
            "data": [
                {"embedding": [0.1, 0.2, 0.3], "index": 0},
                {"embedding": [0.4, 0.5, 0.6], "index": 1},
                {"embedding": [0.7, 0.8, 0.9], "index": 2},
            ],
            "model": "all-MiniLM-L6-v2",
            "usage": {"prompt_tokens": 15},
        }
        response.raise_for_status = MagicMock()
        return response

    @pytest.mark.asyncio
    async def test_embed_single_text(self, mock_response):
        """Test embedding a single text."""
        embedder = Embedder(base_url="http://test:8000/v1", model="test-model")

        with patch.object(embedder._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            result = await embedder.embed("Hello world")

            mock_post.assert_called_once_with(
                "http://test:8000/v1/embeddings",
                json={"model": "test-model", "input": "Hello world"},
            )
            assert result.embedding == [0.1, 0.2, 0.3]
            assert result.model == "all-MiniLM-L6-v2"
            assert result.prompt_tokens == 5

        await embedder.close()

    @pytest.mark.asyncio
    async def test_embed_strips_trailing_slash(self, mock_response):
        """Test that trailing slash in base_url is handled."""
        embedder = Embedder(base_url="http://test:8000/v1/", model="test-model")

        with patch.object(embedder._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            await embedder.embed("Test")

            # Should not have double slash
            mock_post.assert_called_once()
            call_url = mock_post.call_args[0][0]
            assert call_url == "http://test:8000/v1/embeddings"

        await embedder.close()

    @pytest.mark.asyncio
    async def test_embed_batch(self, mock_batch_response):
        """Test embedding multiple texts in batch."""
        embedder = Embedder(base_url="http://test:8000/v1", model="test-model")

        with patch.object(embedder._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_batch_response

            results = await embedder.embed_batch(["text1", "text2", "text3"])

            mock_post.assert_called_once_with(
                "http://test:8000/v1/embeddings",
                json={"model": "test-model", "input": ["text1", "text2", "text3"]},
            )
            assert len(results) == 3
            assert results[0].embedding == [0.1, 0.2, 0.3]
            assert results[1].embedding == [0.4, 0.5, 0.6]
            assert results[2].embedding == [0.7, 0.8, 0.9]
            # Tokens split evenly
            assert results[0].prompt_tokens == 5

        await embedder.close()

    @pytest.mark.asyncio
    async def test_embed_batch_empty_list(self):
        """Test embedding empty list returns empty list."""
        embedder = Embedder(base_url="http://test:8000/v1")

        results = await embedder.embed_batch([])

        assert results == []
        await embedder.close()

    @pytest.mark.asyncio
    async def test_embed_batch_preserves_order(self):
        """Test that batch results are returned in input order."""
        embedder = Embedder(base_url="http://test:8000/v1")

        # Response with shuffled indices
        response = MagicMock()
        response.json.return_value = {
            "data": [
                {"embedding": [0.3], "index": 2},
                {"embedding": [0.1], "index": 0},
                {"embedding": [0.2], "index": 1},
            ],
            "model": "test",
            "usage": {"prompt_tokens": 3},
        }
        response.raise_for_status = MagicMock()

        with patch.object(embedder._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = response

            results = await embedder.embed_batch(["a", "b", "c"])

            # Should be sorted by index
            assert results[0].embedding == [0.1]
            assert results[1].embedding == [0.2]
            assert results[2].embedding == [0.3]

        await embedder.close()

    @pytest.mark.asyncio
    async def test_embed_http_error_propagates(self):
        """Test that HTTP errors are propagated."""
        embedder = Embedder(base_url="http://test:8000/v1")

        response = MagicMock()
        response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Internal Server Error",
            request=MagicMock(),
            response=MagicMock(status_code=500),
        )

        with patch.object(embedder._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = response

            with pytest.raises(httpx.HTTPStatusError):
                await embedder.embed("test")

        await embedder.close()

    @pytest.mark.asyncio
    async def test_embed_missing_usage_defaults_to_zero(self):
        """Test that missing usage info defaults to 0 tokens."""
        embedder = Embedder(base_url="http://test:8000/v1")

        response = MagicMock()
        response.json.return_value = {
            "data": [{"embedding": [0.1], "index": 0}],
            "model": "test",
            # No usage field
        }
        response.raise_for_status = MagicMock()

        with patch.object(embedder._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = response

            result = await embedder.embed("test")
            assert result.prompt_tokens == 0

        await embedder.close()

    @pytest.mark.asyncio
    async def test_context_manager(self, mock_response):
        """Test async context manager usage."""
        async with Embedder(base_url="http://test:8000/v1") as embedder:
            with patch.object(embedder._client, "post", new_callable=AsyncMock) as mock_post:
                mock_post.return_value = mock_response

                result = await embedder.embed("test")
                assert result.embedding == [0.1, 0.2, 0.3]

    @pytest.mark.asyncio
    async def test_close_closes_client(self):
        """Test that close() closes the underlying client."""
        embedder = Embedder(base_url="http://test:8000/v1")

        with patch.object(embedder._client, "aclose", new_callable=AsyncMock) as mock_close:
            await embedder.close()
            mock_close.assert_called_once()
