"""Embedding client for generating vector embeddings via OpenAI-compatible API."""

from dataclasses import dataclass

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

    Usage:
        embedder = Embedder(base_url="http://localhost:8000/v1")
        result = await embedder.embed("User prefers concise explanations")
        embedding = result.embedding  # list[float] of 384 dimensions
        await embedder.close()

    Or as async context manager:
        async with Embedder(base_url="http://localhost:8000/v1") as embedder:
            result = await embedder.embed("User prefers concise explanations")
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model: str = "default",
        timeout: float = 30.0,
    ) -> None:
        """
        Initialize the embedder.

        Args:
            base_url: Base URL for the embedding API (e.g., "http://localhost:8000/v1").
            model: Model name to use for embeddings.
            timeout: Request timeout in seconds.
        """
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._client = httpx.AsyncClient(timeout=timeout)

    @property
    def model(self) -> str:
        """Return the model name used for embeddings."""
        return self._model

    async def close(self) -> None:
        """Close the HTTP client and release resources."""
        await self._client.aclose()

    async def __aenter__(self) -> "Embedder":
        """Enter async context manager."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit async context manager."""
        await self.close()

    async def embed(self, text: str) -> EmbeddingResult:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed.

        Returns:
            EmbeddingResult with embedding vector and metadata.

        Raises:
            httpx.HTTPStatusError: If the API returns an error.
        """
        response = await self._client.post(
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

    async def embed_batch(self, texts: list[str]) -> list[EmbeddingResult]:
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

        response = await self._client.post(
            f"{self._base_url}/embeddings",
            json={"model": self._model, "input": texts},
        )
        response.raise_for_status()
        data = response.json()

        model = data.get("model", self._model)
        prompt_tokens = data.get("usage", {}).get("prompt_tokens", 0)
        tokens_per_text = prompt_tokens // len(texts) if texts else 0

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
