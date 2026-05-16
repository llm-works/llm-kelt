"""Entity embedding adapter - uses core EmbeddingStore for entity vectors."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from appinfra.db.utils import detach_all
from sqlalchemy import select

from .models import Entity
from .store import build_scope_filter

if TYPE_CHECKING:
    from llm_kelt.core.embedding import EmbeddingStore
    from llm_kelt.inference.embedder import Embedder


class EntityEmbeddingAdapter:
    """Embedding operations for KG entities.

    Provides entity-specific interface on top of the core EmbeddingStore.
    Uses entity_type "kg.entity" to namespace embeddings.

    Entities are embedded by concatenating canonical_name + description.

    Example:
        adapter = EntityEmbeddingAdapter(session_factory, store, embedder)

        # Embed an entity
        adapter.embed_entity(entity, "text-embedding-3-small")

        # Search for similar entities
        results = adapter.search_similar(query_embedding, "global", "text-embedding-3-small")

        # Cleanup when entity is deleted
        adapter.delete_embedding(entity_id)
    """

    ENTITY_TYPE = "kg.entity"

    def __init__(
        self,
        session_factory: Callable[[], Any],
        store: EmbeddingStore,
        embedder: Embedder | None = None,
    ) -> None:
        """Initialize EntityEmbeddingAdapter.

        Args:
            session_factory: Callable that returns a context manager for database sessions.
            store: Core EmbeddingStore for vector operations.
            embedder: Optional Embedder for generating embeddings.
        """
        self._session_factory = session_factory
        self._store = store
        self._embedder = embedder

    def _entity_text(self, entity: Entity) -> str:
        """Build text representation of entity for embedding."""
        parts = [entity.canonical_name]
        if entity.description:
            parts.append(entity.description)
        return " ".join(parts)

    def embed_entity(
        self, entity: Entity, model_name: str | None = None, session: Any | None = None
    ) -> None:
        """Generate and store embedding for an entity.

        Args:
            entity: The entity to embed.
            model_name: Embedding model name. If None, uses embedder's default model.
            session: Optional session to use.

        Raises:
            RuntimeError: If no embedder is configured.
        """
        if not self._embedder:
            raise RuntimeError("No embedder configured")

        model = model_name or self._embedder.model
        if model_name is not None and model_name != self._embedder.model:
            raise ValueError(
                f"embedder model {self._embedder.model!r} does not match requested {model_name!r}"
            )

        text = self._entity_text(entity)
        result = self._embedder.embed(text)
        self._store.store(
            entity_type=self.ENTITY_TYPE,
            entity_id=str(entity.id),
            embedding=result.embedding,
            model_name=model,
            session=session,
        )

    def set_embedding(
        self,
        entity_id: int,
        embedding: list[float],
        model_name: str,
    ) -> None:
        """Store a pre-computed embedding for an entity.

        Args:
            entity_id: The entity ID.
            embedding: Pre-computed embedding vector.
            model_name: Embedding model name.
        """
        self._store.store(
            entity_type=self.ENTITY_TYPE,
            entity_id=str(entity_id),
            embedding=embedding,
            model_name=model_name,
        )

    def get_embedding(self, entity_id: int, model_name: str) -> list[float] | None:
        """Get embedding for an entity.

        Args:
            entity_id: The entity ID.
            model_name: Embedding model name.

        Returns:
            The embedding vector, or None if not found.
        """
        return self._store.get(
            entity_type=self.ENTITY_TYPE,
            entity_id=str(entity_id),
            model_name=model_name,
        )

    def delete_embedding(self, entity_id: int) -> int:
        """Delete all embeddings for an entity.

        Args:
            entity_id: The entity ID.

        Returns:
            Number of embeddings deleted.
        """
        return self._store.delete(
            entity_type=self.ENTITY_TYPE,
            entity_id=str(entity_id),
        )

    def search_similar(
        self,
        query_embedding: list[float],
        scope_key: str,
        model_name: str,
        *,
        entity_type: str | None = None,
        limit: int = 10,
    ) -> list[tuple[Entity, float]]:
        """Search for similar entities within a scope.

        Args:
            query_embedding: Query vector.
            scope_key: Scope for hierarchical filtering (includes ancestors up to global).
            model_name: Embedding model name.
            entity_type: Optional entity type filter.
            limit: Maximum results.

        Returns:
            List of (Entity, similarity_score) tuples, ordered by similarity.

        Note:
            Over-fetches 3x to account for scope filtering. May return fewer than
            `limit` results if most embeddings are in scopes outside the query scope.
        """
        raw_results = self._store.search(
            query=query_embedding,
            entity_type=self.ENTITY_TYPE,
            model_name=model_name,
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
