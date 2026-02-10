"""Atomic memory embedding adapter - uses core EmbeddingStore."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

from appinfra.db.utils import detach_all
from sqlalchemy import select

from llm_learn.core.embedding import EmbeddingStore
from llm_learn.core.types import ScoredEntity
from llm_learn.memory.isolation import build_context_filter

from .models import Fact

if TYPE_CHECKING:
    from llm_learn.inference.embedder import Embedder


class EmbeddingAdapter:
    """
    Embedding operations for atomic facts.

    Provides a fact-specific interface on top of the core EmbeddingStore.
    Uses entity_type "atomic.fact" to namespace embeddings.

    Example:
        adapter = EmbeddingAdapter(session_factory, context_key, store, embedder)

        # Embed a fact
        adapter.embed_fact(fact, "text-embedding-3-small")

        # Search for similar facts
        results = adapter.search_similar(query_embedding, "text-embedding-3-small")

        # Cleanup when fact is deleted
        adapter.delete_embedding(fact_id)
    """

    ENTITY_TYPE = "atomic.fact"

    def __init__(
        self,
        session_factory: Callable[[], Any],
        context_key: str | None,
        store: EmbeddingStore,
        embedder: Embedder | None = None,
    ) -> None:
        """
        Initialize EmbeddingAdapter.

        Args:
            session_factory: Callable that returns a context manager for database sessions.
            context_key: Profile ID (32-char hash) to scope operations to.
            store: Core EmbeddingStore for vector operations.
            embedder: Optional Embedder for generating embeddings.
        """
        self._session_factory = session_factory
        self._context_key = context_key
        self._store = store
        self._embedder = embedder

    def embed_fact(self, fact: Fact, model_name: str | None = None) -> None:
        """
        Generate and store embedding for a fact.

        Args:
            fact: The fact to embed (uses fact.content).
            model_name: Embedding model name. If None, uses embedder's default model.

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
        result = self._embedder.embed(fact.content)
        self._store.store(
            entity_type=self.ENTITY_TYPE,
            entity_id=str(fact.id),
            embedding=result.embedding,
            model_name=model,
        )

    def set_embedding(
        self,
        fact_id: int,
        embedding: list[float],
        model_name: str,
    ) -> None:
        """
        Store a pre-computed embedding for a fact.

        Args:
            fact_id: The fact ID.
            embedding: Pre-computed embedding vector.
            model_name: Embedding model name.
        """
        self._store.store(
            entity_type=self.ENTITY_TYPE,
            entity_id=str(fact_id),
            embedding=embedding,
            model_name=model_name,
        )

    def get_embedding(self, fact_id: int, model_name: str) -> list[float] | None:
        """
        Get embedding for a fact.

        Args:
            fact_id: The fact ID.
            model_name: Embedding model name.

        Returns:
            Embedding vector if found, None otherwise.
        """
        return self._store.get(self.ENTITY_TYPE, str(fact_id), model_name)

    def _hydrate_facts(
        self,
        fact_ids: list[int],
        fact_type: str | None,
        categories: list[str] | None,
    ) -> list[Fact]:
        """Fetch facts from DB with optional filters."""
        with self._session_factory() as session:
            stmt = select(Fact).where(
                Fact.id.in_(fact_ids),
                Fact.active == True,  # noqa: E712
            )

            # Apply context filtering with glob pattern support
            context_filter = build_context_filter(self._context_key, Fact.context_key)
            if context_filter is not None:
                stmt = stmt.where(context_filter)

            if fact_type:
                stmt = stmt.where(Fact.type == fact_type)
            if categories:
                stmt = stmt.where(Fact.category.in_(categories))

            facts = list(session.scalars(stmt).all())
            return cast(list[Fact], detach_all(facts, session))

    def search_similar(
        self,
        query: list[float],
        model_name: str,
        *,
        top_k: int = 10,
        min_similarity: float = 0.0,
        fact_type: str | None = None,
        categories: list[str] | None = None,
    ) -> list[ScoredEntity[Fact]]:
        """Search for facts similar to a query embedding."""
        # Over-fetch if we need to filter
        fetch_k = top_k * 2 if (fact_type or categories) else top_k

        results = self._store.search(
            query=query,
            entity_type=self.ENTITY_TYPE,
            model_name=model_name,
            top_k=fetch_k,
            min_similarity=min_similarity,
        )
        if not results:
            return []

        # Build score map and hydrate facts
        fact_ids = [int(entity_id) for entity_id, _ in results]
        score_map = {int(entity_id): score for entity_id, score in results}
        facts = self._hydrate_facts(fact_ids, fact_type, categories)

        # Build scored results, sorted by similarity
        scored = [ScoredEntity(entity=f, score=score_map[f.id]) for f in facts if f.id in score_map]
        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:top_k]

    def delete_embedding(self, fact_id: int) -> int:
        """
        Delete embeddings when a fact is deleted.

        Should be called when a fact is hard-deleted to prevent orphan embeddings.

        Args:
            fact_id: The fact ID.

        Returns:
            Number of embeddings deleted.
        """
        return self._store.delete(self.ENTITY_TYPE, str(fact_id))

    def has_embedding(self, fact_id: int, model_name: str) -> bool:
        """
        Check if a fact has an embedding for a specific model.

        Args:
            fact_id: The fact ID.
            model_name: Embedding model name.

        Returns:
            True if embedding exists.
        """
        return self._store.exists(self.ENTITY_TYPE, str(fact_id), model_name)

    def list_without_embeddings(
        self,
        model_name: str,
        *,
        fact_type: str | None = None,
        limit: int = 100,
    ) -> list[Fact]:
        """
        Find facts that need embeddings for a specific model.

        Useful for batch embedding generation.

        Args:
            model_name: Embedding model name.
            fact_type: Optional fact type filter.
            limit: Maximum facts to return.

        Returns:
            List of facts without embeddings for the model.
        """
        with self._session_factory() as session:
            # Get candidate facts
            stmt = select(Fact).where(
                Fact.active == True,  # noqa: E712
            )

            # Apply context filtering with glob pattern support
            context_filter = build_context_filter(self._context_key, Fact.context_key)
            if context_filter is not None:
                stmt = stmt.where(context_filter)

            if fact_type:
                stmt = stmt.where(Fact.type == fact_type)

            # Fetch more than limit to find enough facts without embeddings
            # Needs to be high enough to find limit results after filtering
            facts = list(session.scalars(stmt.limit(limit * 10)).all())

            # Filter to those without embeddings
            fact_ids = [str(f.id) for f in facts]
            missing_ids = set(self._store.list_missing(self.ENTITY_TYPE, fact_ids, model_name))

            result = [f for f in facts if str(f.id) in missing_ids][:limit]
            return cast(list[Fact], detach_all(result, session))

    def count(self, model_name: str | None = None) -> int:
        """
        Count embeddings for atomic facts.

        Args:
            model_name: Optional model filter.

        Returns:
            Total count.
        """
        return self._store.count(entity_type=self.ENTITY_TYPE, model_name=model_name)
