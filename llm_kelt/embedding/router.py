"""Embedding router - routes operations to format-specific stores."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from .store.f16 import Float16Store
from .store.f32 import Float32Store
from .store.i4 import Int4Store
from .store.i8 import Int8Store
from .types import EmbeddingConfig, QuantizationFormat

if TYPE_CHECKING:
    from .store.base import EmbeddingStoreProtocol


class EmbeddingRouter:
    """Routes embedding operations based on format configuration.

    Manages multiple format-specific stores and handles two-phase search
    with optional reranking for quantized formats.

    Example:
        router = EmbeddingRouter(session_factory, dimensions=384)

        # Store in multiple formats
        config = EmbeddingConfig(
            context_key="profile_abc",
            store_formats=[QuantizationFormat.F32, QuantizationFormat.I8],
            search_format=QuantizationFormat.I8,
            rerank_format=QuantizationFormat.F32,
        )
        router.store("atomic.fact", "42", embedding, "minilm", config)

        # Search with reranking
        results = router.search(query, "atomic.fact", "minilm", top_k=10, config=config)
    """

    def __init__(self, session_factory: Callable[[], Any], dimensions: int) -> None:
        self._session_factory = session_factory
        self._dimensions = dimensions
        self._stores: dict[QuantizationFormat, EmbeddingStoreProtocol] = {
            QuantizationFormat.F32: Float32Store(session_factory, dimensions),
            QuantizationFormat.F16: Float16Store(session_factory, dimensions),
            QuantizationFormat.I8: Int8Store(session_factory, dimensions),
            QuantizationFormat.I4: Int4Store(session_factory, dimensions),
        }

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def get_store(self, fmt: QuantizationFormat) -> EmbeddingStoreProtocol:
        """Get the store for a specific format."""
        return self._stores[fmt]

    def store(
        self,
        entity_type: str,
        entity_id: str,
        embedding: list[float],
        model_name: str,
        config: EmbeddingConfig,
        session: Any | None = None,
    ) -> None:
        """Store embedding in all configured formats.

        Args:
            entity_type: Type prefix (e.g., "atomic.fact").
            entity_id: Entity identifier.
            embedding: Float32 embedding vector.
            model_name: Embedding model name.
            config: Embedding configuration.
            session: Optional session for transaction participation.
        """
        for fmt in config.store_formats:
            store = self._stores[fmt]
            store.store(entity_type, entity_id, embedding, model_name, session=session)

    def search(
        self,
        query: list[float],
        entity_type: str,
        model_name: str,
        top_k: int,
        config: EmbeddingConfig,
        min_similarity: float = 0.0,
        entity_id_subquery: Any | None = None,
    ) -> list[tuple[str, float]]:
        """Two-phase search with optional reranking.

        Phase 1: Retrieve candidates from search_format store (oversampled if reranking).
        Phase 2: If rerank_format is set, re-score candidates using higher-precision embeddings.

        Args:
            query: Query embedding vector (float32).
            entity_type: Type prefix to search within.
            model_name: Embedding model to search.
            top_k: Maximum results to return.
            config: Embedding configuration.
            min_similarity: Minimum similarity threshold.
            entity_id_subquery: Optional subquery for pre-filtering.

        Returns:
            List of (entity_id, similarity) tuples, ordered by similarity descending.
        """
        search_store = self._stores[config.search_format]

        if config.rerank_format:
            fetch_k = top_k * config.rerank_oversample
        else:
            fetch_k = top_k

        candidates = search_store.search(
            query=query,
            entity_type=entity_type,
            model_name=model_name,
            top_k=fetch_k,
            min_similarity=0.0,
            entity_id_subquery=entity_id_subquery,
        )

        if config.rerank_format and len(candidates) > top_k:
            candidates = self._rerank(
                query=query,
                candidates=candidates,
                rerank_format=config.rerank_format,
                entity_type=entity_type,
                model_name=model_name,
            )

        filtered = [(eid, sim) for eid, sim in candidates if sim >= min_similarity]
        return filtered[:top_k]

    def _rerank(
        self,
        query: list[float],
        candidates: list[tuple[str, float]],
        rerank_format: QuantizationFormat,
        entity_type: str,
        model_name: str,
    ) -> list[tuple[str, float]]:
        """Re-score candidates using higher-precision embeddings."""
        rerank_store = self._stores[rerank_format]
        reranked = []

        for entity_id, _ in candidates:
            embedding = rerank_store.get(entity_type, entity_id, model_name)
            if embedding is not None:
                sim = self._cosine_similarity(query, embedding)
                reranked.append((entity_id, sim))

        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def get(
        self,
        entity_type: str,
        entity_id: str,
        model_name: str,
        fmt: QuantizationFormat = QuantizationFormat.F32,
    ) -> list[float] | None:
        """Get embedding from specified format store (dequantized to float32).

        Args:
            entity_type: Type prefix.
            entity_id: Entity identifier.
            model_name: Embedding model name.
            fmt: Format to retrieve from.

        Returns:
            Float32 embedding vector if found, None otherwise.
        """
        return self._stores[fmt].get(entity_type, entity_id, model_name)

    def delete(
        self,
        entity_type: str,
        entity_id: str,
        config: EmbeddingConfig,
        session: Any | None = None,
    ) -> int:
        """Delete embeddings from all configured format stores.

        Args:
            entity_type: Type prefix.
            entity_id: Entity identifier.
            config: Embedding configuration.
            session: Optional session for transaction participation.

        Returns:
            Total number of embeddings deleted across all formats.
        """
        total = 0
        for fmt in config.store_formats:
            store = self._stores[fmt]
            total += store.delete(entity_type, entity_id, session=session)
        return total

    def exists(
        self,
        entity_type: str,
        entity_id: str,
        model_name: str,
        fmt: QuantizationFormat = QuantizationFormat.F32,
    ) -> bool:
        """Check if embedding exists in specified format store."""
        return self._stores[fmt].exists(entity_type, entity_id, model_name)

    def count(
        self,
        fmt: QuantizationFormat,
        entity_type: str | None = None,
        model_name: str | None = None,
    ) -> int:
        """Count embeddings in specified format store."""
        return self._stores[fmt].count(entity_type, model_name)
