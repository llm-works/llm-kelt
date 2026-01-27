"""Embedding client for generating vector embeddings via OpenAI-compatible API."""

from dataclasses import dataclass
from typing import Self

import httpx


@dataclass
class EmbeddingResult:
    """Result from embedding generation."""

    embedding: list[float]
    model: str
    prompt_tokens: int


class Embedder:
    """
    Client for generating embeddings via OpenAI-compatible API.

    Connects to an embedding endpoint (e.g., llm-infer) to generate
    vector representations for semantic search.

    Sync usage (default):
        with Embedder(base_url="http://localhost:8001/v1") as embedder:
            result = embedder.embed("User prefers concise explanations")
            embedding = result.embedding  # list[float]

    Async usage:
        async with Embedder(base_url="http://localhost:8001/v1") as embedder:
            result = await embedder.embed_async("User prefers concise explanations")
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8001/v1",
        model: str = "default",
        timeout: float = 30.0,
    ) -> None:
        """
        Initialize the embedder.

        Args:
            base_url: Base URL for the embedding API (e.g., "http://localhost:8001/v1").
            model: Model name to use for embeddings.
            timeout: Request timeout in seconds.
        """
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout
        self._client = httpx.Client(timeout=timeout)
        self._async_client: httpx.AsyncClient | None = None

    @property
    def model(self) -> str:
        """Return the model name used for embeddings."""
        return self._model

    def _get_async_client(self) -> httpx.AsyncClient:
        """Get or create the async HTTP client (lazy initialization)."""
        if self._async_client is None:
            self._async_client = httpx.AsyncClient(timeout=self._timeout)
        return self._async_client

    # -------------------------------------------------------------------------
    # Resource management
    # -------------------------------------------------------------------------

    def close(self) -> None:
        """Close the sync HTTP client."""
        self._client.close()

    async def close_async(self) -> None:
        """Close all HTTP clients (sync and async)."""
        self._client.close()
        if self._async_client is not None:
            await self._async_client.aclose()
            self._async_client = None

    def __enter__(self) -> Self:
        """Enter sync context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit sync context manager."""
        self.close()

    async def __aenter__(self) -> Self:
        """Enter async context manager."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit async context manager."""
        await self.close_async()

    # -------------------------------------------------------------------------
    # Sync API (default)
    # -------------------------------------------------------------------------

    def embed(self, text: str) -> EmbeddingResult:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed.

        Returns:
            EmbeddingResult with embedding vector and metadata.

        Raises:
            httpx.HTTPStatusError: If the API returns an error.
        """
        response = self._client.post(
            f"{self._base_url}/embeddings",
            json={"model": self._model, "input": text},
        )
        response.raise_for_status()
        data = response.json()

        return EmbeddingResult(
            embedding=data["data"][0]["embedding"],
            model=data.get("model", self._model),
            prompt_tokens=data.get("usage", {}).get("prompt_tokens", 0),
        )

    def embed_batch(self, texts: list[str]) -> list[EmbeddingResult]:
        """
        Generate embeddings for multiple texts in a single request.

        Args:
            texts: List of texts to embed.

        Returns:
            List of EmbeddingResult, one per input text.

        Raises:
            httpx.HTTPStatusError: If the API returns an error.
        """
        if not texts:
            return []

        response = self._client.post(
            f"{self._base_url}/embeddings",
            json={"model": self._model, "input": texts},
        )
        response.raise_for_status()
        data = response.json()

        return self._parse_batch_response(data, len(texts))

    # -------------------------------------------------------------------------
    # Async API
    # -------------------------------------------------------------------------

    async def embed_async(self, text: str) -> EmbeddingResult:
        """
        Generate embedding for a single text (async version).

        Args:
            text: Text to embed.

        Returns:
            EmbeddingResult with embedding vector and metadata.

        Raises:
            httpx.HTTPStatusError: If the API returns an error.
        """
        client = self._get_async_client()
        response = await client.post(
            f"{self._base_url}/embeddings",
            json={"model": self._model, "input": text},
        )
        response.raise_for_status()
        data = response.json()

        return EmbeddingResult(
            embedding=data["data"][0]["embedding"],
            model=data.get("model", self._model),
            prompt_tokens=data.get("usage", {}).get("prompt_tokens", 0),
        )

    async def embed_batch_async(self, texts: list[str]) -> list[EmbeddingResult]:
        """
        Generate embeddings for multiple texts in a single request (async version).

        Args:
            texts: List of texts to embed.

        Returns:
            List of EmbeddingResult, one per input text.

        Raises:
            httpx.HTTPStatusError: If the API returns an error.
        """
        if not texts:
            return []

        client = self._get_async_client()
        response = await client.post(
            f"{self._base_url}/embeddings",
            json={"model": self._model, "input": texts},
        )
        response.raise_for_status()
        data = response.json()

        return self._parse_batch_response(data, len(texts))

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _parse_batch_response(self, data: dict, num_texts: int) -> list[EmbeddingResult]:
        """Parse batch embedding response into EmbeddingResult list."""
        model = data.get("model", self._model)
        prompt_tokens = data.get("usage", {}).get("prompt_tokens", 0)
        tokens_per_text = prompt_tokens // num_texts if num_texts else 0

        results = []
        for item in sorted(data["data"], key=lambda x: x["index"]):
            results.append(
                EmbeddingResult(
                    embedding=item["embedding"],
                    model=model,
                    prompt_tokens=tokens_per_text,
                )
            )
        return results
