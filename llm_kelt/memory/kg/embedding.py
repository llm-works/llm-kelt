"""Entity embedding adapter - uses embedding.Factory for dynamic dimension routing."""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from appinfra.db.utils import detach_all
from sqlalchemy import select

from llm_kelt.embedding import Config as EmbeddingConfig
from llm_kelt.embedding import Factory as EmbeddingFactory
from llm_kelt.embedding import StoreClient as EmbeddingStoreClient
from llm_kelt.embedding.types import QuantizationFormat

from .models import Entity
from .store import build_scope_filter

if TYPE_CHECKING:
    from llm_infer.client import EmbeddingClient


class EntityEmbeddingAdapter:
    """Embedding operations for KG entities with dynamic dimension routing.

    Routes embeddings to the correct table based on their dimensions.
    Uses entity_type "kg.entity" to namespace embeddings.

    Entities are embedded by concatenating canonical_name + description.

    Example:
        adapter = EntityEmbeddingAdapter(
            session_factory=session_factory,
            factory=EmbeddingFactory(),
            format=QuantizationFormat.F16,
            embedder=embedder,
        )

        # Embed an entity (auto-routes to correct dimension table)
        adapter.embed_entity(entity)

        # Search for similar entities
        results = adapter.search_similar(query_embedding, "global", "text-embedding-3-small")

        # Cleanup when entity is deleted
        adapter.delete_embedding(entity_id)
    """

    ENTITY_TYPE = "kg.entity"

    def __init__(
        self,
        session_factory: Callable[[], Any],
        factory: EmbeddingFactory,
        format: QuantizationFormat,
        embedder: EmbeddingClient | None = None,
        default_dimensions: int | None = None,
    ) -> None:
        """Initialize EntityEmbeddingAdapter with dynamic dimension routing.

        Args:
            session_factory: Callable that returns a context manager for database sessions.
            factory: EmbeddingFactory for creating dimension-specific stores.
            format: Quantization format to use (F32, F16, I8, I4).
            embedder: Optional EmbeddingClient for generating embeddings via HTTP.
            default_dimensions: Default dimensions for operations when not specified.
        """
        self._session_factory = session_factory
        self._factory = factory
        self._format = format
        self._embedder = embedder
        self._default_dimensions = default_dimensions
        self._stores: dict[int, EmbeddingStoreClient] = {}
        self._stores_lock = threading.Lock()

    def _get_store(self, dimensions: int) -> EmbeddingStoreClient:
        """Get or create a store for the given dimensions (thread-safe)."""
        if dimensions in self._stores:
            return self._stores[dimensions]
        with self._stores_lock:
            if dimensions not in self._stores:
                config = EmbeddingConfig(
                    context_key="_kg",
                    format=self._format,
                    dimensions=dimensions,
                )
                self._stores[dimensions] = self._factory.create(self._session_factory, config)
            return self._stores[dimensions]

    def _entity_text(self, entity: Entity) -> str:
        """Build text representation of entity for embedding."""
        parts = [entity.canonical_name]
        if entity.description:
            parts.append(entity.description)
        return " ".join(parts)

    def embed_entity(
        self, entity: Entity, model: str | None = None, session: Any | None = None
    ) -> None:
        """Generate and store embedding for an entity.

        Routes to the correct dimension table based on embedding output.

        Args:
            entity: The entity to embed.
            model: Embedding model name. If None, uses embedder's default model.
            session: Optional session to use.

        Raises:
            RuntimeError: If no embedder is configured.
        """
        if not self._embedder:
            raise RuntimeError("No embedder configured")

        resolved_model = model or self._embedder.model
        if model is not None and model != self._embedder.model:
            raise ValueError(
                f"embedder model {self._embedder.model!r} does not match requested {model!r}"
            )

        text = self._entity_text(entity)
        result = self._embedder.embed(text)
        store = self._get_store(len(result.embedding))
        store.store(
            entity_type=self.ENTITY_TYPE,
            entity_id=str(entity.id),
            embedding=result.embedding,
            model=resolved_model,
            session=session,
        )

    def set_embedding(
        self,
        entity_id: int,
        embedding: list[float],
        model: str,
    ) -> None:
        """Store a pre-computed embedding for an entity.

        Routes to the correct table based on embedding dimensions.

        Args:
            entity_id: The entity ID.
            embedding: Pre-computed embedding vector.
            model: Embedding model name.
        """
        store = self._get_store(len(embedding))
        store.store(
            entity_type=self.ENTITY_TYPE,
            entity_id=str(entity_id),
            embedding=embedding,
            model=model,
        )

    def get_embedding(
        self, entity_id: int, model: str, dimensions: int | None = None
    ) -> list[float] | None:
        """Get embedding for an entity.

        Args:
            entity_id: The entity ID.
            model: Embedding model name.
            dimensions: Embedding dimensions (determines which table to query).
                If None, uses default_dimensions.

        Returns:
            The embedding vector, or None if not found.

        Raises:
            ValueError: If dimensions is None and no default_dimensions configured.
        """
        dims = dimensions if dimensions is not None else self._default_dimensions
        if dims is None:
            raise ValueError("dimensions required when no default_dimensions configured")
        store = self._get_store(dims)
        return store.get(
            entity_type=self.ENTITY_TYPE,
            entity_id=str(entity_id),
            model=model,
        )

    def delete_embedding(self, entity_id: int, *, dimensions: int | None = None) -> int:
        """Delete embeddings for an entity.

        Args:
            entity_id: The entity ID.
            dimensions: Embedding dimensions to delete from. If None, uses
                default_dimensions. Falls back to cached stores if neither set.

        Returns:
            Number of embeddings deleted.
        """
        dims = dimensions if dimensions is not None else self._default_dimensions
        if dims is not None:
            store = self._get_store(dims)
            return store.delete(entity_type=self.ENTITY_TYPE, entity_id=str(entity_id))
        total = 0
        for store in self._stores.values():
            total += store.delete(
                entity_type=self.ENTITY_TYPE,
                entity_id=str(entity_id),
            )
        return total

    def search_similar(
        self,
        query_embedding: list[float],
        scope_key: str,
        model: str,
        *,
        entity_type: str | None = None,
        limit: int = 10,
    ) -> list[tuple[Entity, float]]:
        """Search for similar entities within a scope.

        Routes to the correct dimension table based on query vector length.

        Args:
            query_embedding: Query vector.
            scope_key: Scope for hierarchical filtering (includes ancestors up to global).
            model: Embedding model name.
            entity_type: Optional entity type filter.
            limit: Maximum results.

        Returns:
            List of (Entity, similarity_score) tuples, ordered by similarity.

        Note:
            Over-fetches 3x to account for scope filtering. May return fewer than
            `limit` results if most embeddings are in scopes outside the query scope.
        """
        store = self._get_store(len(query_embedding))
        raw_results = store.search(
            query=query_embedding,
            entity_type=self.ENTITY_TYPE,
            model=model,
            top_k=limit * 3,  # Over-fetch to account for scope filtering
        )

        if not raw_results:
            return []

        # raw_results is list[tuple[str, float]] - (entity_id, similarity)
        entity_ids = [int(eid) for eid, _ in raw_results]
        score_map = {int(eid): sim for eid, sim in raw_results}

        with self._session_factory() as session:
            stmt = select(Entity).where(
                Entity.id.in_(entity_ids),
                build_scope_filter(scope_key, Entity.scope_key),
            )
            if entity_type:
                stmt = stmt.where(Entity.entity_type == entity_type)

            entities = list(session.scalars(stmt))
            detach_all(entities, session)

        results = [(e, score_map[e.id]) for e in entities if e.id in score_map]
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]

    def embed_text(self, text: str) -> list[float]:
        """Generate embedding for arbitrary text (for queries).

        Args:
            text: Text to embed.

        Returns:
            Embedding vector.

        Raises:
            RuntimeError: If no embedder is configured.
        """
        if not self._embedder:
            raise RuntimeError("No embedder configured")
        return self._embedder.embed(text).embedding
