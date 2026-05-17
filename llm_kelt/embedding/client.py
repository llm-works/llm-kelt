"""Embeddings client - high-level API for embedding operations."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from llm_kelt.core.errors import ValidationError

from .types import Calibration, Config

if TYPE_CHECKING:
    from .store.base import EmbeddingStoreProtocol


def _validate_embedding(embedding: list[float] | None) -> None:
    """Validate an embedding vector.

    Args:
        embedding: The embedding vector to validate.

    Raises:
        ValidationError: If embedding is empty, contains non-numeric, or non-finite values.
    """
    if not embedding:
        raise ValidationError("embedding cannot be empty")

    for i, val in enumerate(embedding):
        if not isinstance(val, (int, float)):
            raise ValidationError(
                f"embedding values must be numeric, got {type(val).__name__} at index {i}"
            )
        if math.isnan(val) or math.isinf(val):
            raise ValidationError(f"embedding values must be finite, got {val} at index {i}")


class Client:
    """High-level client for embedding storage and search.

    Each client uses ONE format and ONE dimension. Want multiple formats?
    Create multiple clients via Factory with different configs.

    Example:
        from llm_kelt.embedding import Factory, Config, QuantizationFormat

        factory = Factory()
        config = Config(
            context_key="my-context",
            format=QuantizationFormat.F16,
            dimensions=384,
        )
        client = factory.create(session_factory, config)

        # Store embedding
        client.store("atomic.fact", "42", embedding, "minilm")

        # Search
        results = client.search(query, "atomic.fact", "minilm", top_k=10)
    """

    def __init__(self, config: Config, store: EmbeddingStoreProtocol) -> None:
        """Initialize embedding client.

        Args:
            config: Embedding configuration (format, dimensions).
            store: The underlying store (created by Factory).
        """
        self._config = config
        self._store = store
        self._table_ensured = False

    def _ensure_table(self) -> None:
        """Ensure table exists (only runs once)."""
        if not self._table_ensured:
            self._store.ensure_table()
            self._table_ensured = True

    @property
    def config(self) -> Config:
        """Embedding configuration."""
        return self._config

    @property
    def dimensions(self) -> int:
        """Vector dimensions."""
        return self._config.dimensions

    def store(
        self,
        entity_type: str,
        entity_id: str,
        embedding: list[float],
        model_name: str,
        calibration: Calibration | None = None,
        session: Any | None = None,
    ) -> None:
        """Store embedding.

        Args:
            entity_type: Type prefix (e.g., "atomic.fact", "kg.entity").
            entity_id: Entity identifier.
            embedding: Float32 embedding vector.
            model_name: Embedding model name.
            calibration: Optional calibration for quantized formats (I8/I4).
            session: Optional session for transaction participation.

        Raises:
            ValidationError: If embedding is empty or contains non-finite values.
        """
        _validate_embedding(embedding)
        self._ensure_table()
        self._store.store(
            entity_type, entity_id, embedding, model_name, calibration, session=session
        )

    def search(
        self,
        query: list[float],
        entity_type: str,
        model_name: str,
        top_k: int = 10,
        min_similarity: float = 0.0,
        entity_id_subquery: Any | None = None,
    ) -> list[tuple[str, float]]:
        """Search for similar embeddings.

        Args:
            query: Query embedding vector (float32).
            entity_type: Type prefix to search within.
            model_name: Embedding model to search.
            top_k: Maximum results to return.
            min_similarity: Minimum similarity threshold.
            entity_id_subquery: Optional subquery for pre-filtering.

        Returns:
            List of (entity_id, similarity) tuples, ordered by similarity descending.
        """
        self._ensure_table()
        results = self._store.search(
            query=query,
            entity_type=entity_type,
            model_name=model_name,
            top_k=top_k,
            min_similarity=min_similarity,
            entity_id_subquery=entity_id_subquery,
        )
        return results

    def get(
        self,
        entity_type: str,
        entity_id: str,
        model_name: str,
    ) -> list[float] | None:
        """Get embedding for an entity (dequantized to float32).

        Args:
            entity_type: Type prefix.
            entity_id: Entity identifier.
            model_name: Embedding model name.

        Returns:
            Float32 embedding vector if found, None otherwise.
        """
        self._ensure_table()
        return self._store.get(entity_type, entity_id, model_name)

    def delete(
        self,
        entity_type: str,
        entity_id: str,
        session: Any | None = None,
    ) -> int:
        """Delete embedding.

        Args:
            entity_type: Type prefix.
            entity_id: Entity identifier.
            session: Optional session for transaction participation.

        Returns:
            Number of embeddings deleted (0 or 1).
        """
        self._ensure_table()
        return self._store.delete(entity_type, entity_id, session=session)

    def exists(
        self,
        entity_type: str,
        entity_id: str,
        model_name: str,
    ) -> bool:
        """Check if embedding exists.

        Args:
            entity_type: Type prefix.
            entity_id: Entity identifier.
            model_name: Embedding model name.

        Returns:
            True if embedding exists.
        """
        self._ensure_table()
        return self._store.exists(entity_type, entity_id, model_name)

    def count(
        self,
        entity_type: str | None = None,
        model_name: str | None = None,
    ) -> int:
        """Count embeddings.

        Args:
            entity_type: Optional type filter.
            model_name: Optional model filter.

        Returns:
            Total count of matching embeddings.
        """
        self._ensure_table()
        return self._store.count(entity_type, model_name)

    def list_missing(
        self,
        entity_type: str,
        entity_ids: list[str],
        model_name: str,
    ) -> list[str]:
        """Find entity IDs that don't have embeddings.

        Useful for batch embedding generation to identify which entities
        still need embeddings.

        Args:
            entity_type: Type prefix.
            entity_ids: List of entity IDs to check.
            model_name: Embedding model name.

        Returns:
            List of entity IDs missing embeddings.
        """
        if not entity_ids:
            return []

        self._ensure_table()
        existing = self._store.list_existing(entity_type, entity_ids, model_name)
        return [eid for eid in entity_ids if eid not in existing]

    def ensure_table(self) -> None:
        """Explicitly ensure table exists.

        Normally table is created on first use. Call this to pre-create
        before any operations.
        """
        self._store.ensure_table()
        self._table_ensured = True
