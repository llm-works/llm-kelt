"""Atomic memory embedding adapter - uses embedding.StoreClient."""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

from appinfra.db.utils import detach_all
from sqlalchemy import String, and_, select
from sqlalchemy import cast as sa_cast
from sqlalchemy.orm import Session
from sqlalchemy.sql.elements import ColumnElement

from llm_kelt.core.types import ScoredEntity
from llm_kelt.embedding import Config as EmbeddingConfig
from llm_kelt.embedding import Factory as EmbeddingFactory
from llm_kelt.embedding import StoreClient as EmbeddingStoreClient
from llm_kelt.embedding.types import QuantizationFormat
from llm_kelt.memory.isolation import build_context_filter

from .models import Fact

if TYPE_CHECKING:
    from llm_infer.client import EmbeddingClient


class EmbeddingFilter:
    """Fluent builder for embedding similarity search filters.

    Combines convenience methods for common filters with raw SQLAlchemy
    clause support for complex joins and subqueries.

    Examples:
        # Simple filters
        f = EmbeddingFilter().fact_type("solution").categories("joke", "riddle")

        # Complex join filter
        f = EmbeddingFilter().where(
            Fact.id.in_(
                select(ModelUsage.fact_id).where(ModelUsage.model_name.not_ilike('%haiku%'))
            )
        )

        # Combined
        f = EmbeddingFilter().fact_type("solution").where(custom_clause)
    """

    def __init__(self) -> None:
        """Initialize an empty EmbeddingFilter."""
        self._fact_type: str | None = None
        self._categories: list[str] | None = None
        self._clauses: list[ColumnElement[bool]] = []

    def fact_type(self, t: str) -> EmbeddingFilter:
        """Filter by atomic fact type (assertion, solution, prediction, etc.)."""
        self._fact_type = t
        return self

    def categories(self, *cats: str) -> EmbeddingFilter:
        """Filter by fact category (must match one of the provided categories)."""
        self._categories = list(cats)
        return self

    def where(self, clause: ColumnElement[bool]) -> EmbeddingFilter:
        """Add a raw SQLAlchemy clause for complex filtering.

        Can be called multiple times - clauses are ANDed together.
        """
        self._clauses.append(clause)
        return self

    def build(self) -> ColumnElement[bool] | None:
        """Build the combined SQLAlchemy clause."""
        parts: list[ColumnElement[bool]] = []

        if self._fact_type:
            parts.append(Fact.type == self._fact_type)
        if self._categories:
            parts.append(Fact.category.in_(self._categories))
        parts.extend(self._clauses)

        if not parts:
            return None
        return and_(*parts) if len(parts) > 1 else parts[0]

    def __bool__(self) -> bool:
        """Return True if any filters are set."""
        return bool(self._fact_type or self._categories or self._clauses)

    def __repr__(self) -> str:
        """Return string representation for debugging."""
        parts = []
        if self._fact_type:
            parts.append(f"fact_type={self._fact_type!r}")
        if self._categories:
            parts.append(f"categories={self._categories!r}")
        if self._clauses:
            parts.append(f"clauses={len(self._clauses)}")
        return f"EmbeddingFilter({', '.join(parts)})"


class EmbeddingAdapter:
    """
    Embedding operations for atomic facts with dynamic dimension routing.

    Routes embeddings to the correct table based on their dimensions.
    Uses entity_type "atomic.fact" to namespace embeddings.

    Example:
        adapter = EmbeddingAdapter(
            session_factory=session_factory,
            context_key=context_key,
            factory=EmbeddingFactory(),
            format=QuantizationFormat.F16,
            embedder=embedder,
        )

        # Embed a fact (auto-routes to correct dimension table)
        adapter.embed_fact(fact, dimensions=256)

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
        factory: EmbeddingFactory,
        format: QuantizationFormat,
        embedder: EmbeddingClient | None = None,
        default_dimensions: int | None = None,
    ) -> None:
        """
        Initialize EmbeddingAdapter with dynamic dimension routing.

        Args:
            session_factory: Callable that returns a context manager for database sessions.
            context_key: Profile ID (32-char hash) to scope operations to.
            factory: EmbeddingFactory for creating dimension-specific stores.
            format: Quantization format to use (F32, F16, I8, I4).
            embedder: Optional EmbeddingClient for generating embeddings via HTTP.
            default_dimensions: Default dimensions for embed_fact when not specified.
        """
        self._session_factory = session_factory
        self._context_key = context_key
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
                    context_key=self._context_key or "_default",
                    format=self._format,
                    dimensions=dimensions,
                )
                self._stores[dimensions] = self._factory.create(self._session_factory, config)
            return self._stores[dimensions]

    def embed_fact(
        self,
        fact: Fact,
        model: str | None = None,
        dimensions: int | None = None,
        session: Any | None = None,
    ) -> None:
        """
        Generate and store embedding for a fact.

        Args:
            fact: The fact to embed (uses fact.content).
            model: Embedding model name. If None, uses embedder's default model.
            dimensions: Output dimensions. If None, uses adapter default or embedder config.
            session: Optional session to use. If None, creates new session and commits.
                     If provided, uses existing session without committing (caller controls).

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
        dims = dimensions if dimensions is not None else self._default_dimensions
        result = self._embedder.embed(fact.content, dimensions=dims)
        store = self._get_store(len(result.embedding))
        store.store(
            entity_type=self.ENTITY_TYPE,
            entity_id=str(fact.id),
            embedding=result.embedding,
            model=resolved_model,
            session=session,
        )

    def set_embedding(
        self,
        fact_id: int,
        embedding: list[float],
        model: str,
    ) -> None:
        """
        Store a pre-computed embedding for a fact.

        Routes to the correct table based on embedding dimensions.

        Args:
            fact_id: The fact ID.
            embedding: Pre-computed embedding vector.
            model: Embedding model name.
        """
        store = self._get_store(len(embedding))
        store.store(
            entity_type=self.ENTITY_TYPE,
            entity_id=str(fact_id),
            embedding=embedding,
            model=model,
        )

    def get_embedding(self, fact_id: int, model: str, dimensions: int) -> list[float] | None:
        """
        Get embedding for a fact.

        Args:
            fact_id: The fact ID.
            model: Embedding model name.
            dimensions: Embedding dimensions (determines which table to query).

        Returns:
            Embedding vector if found, None otherwise.
        """
        store = self._get_store(dimensions)
        return store.get(self.ENTITY_TYPE, str(fact_id), model)

    def _build_entity_id_subquery(
        self,
        effective_filter: EmbeddingFilter | None,
    ) -> Any:
        """Build subquery that selects fact IDs matching all filter criteria.

        Returns a SQLAlchemy subquery selecting Fact.id cast to string,
        constrained by: active=True, context_key filter, and EmbeddingFilter.
        This subquery is passed to EmbeddingStore.search() for pre-filtering
        the vector search.
        """
        stmt = select(sa_cast(Fact.id, String)).where(Fact.active == True)  # noqa: E712

        # Apply context filtering
        context_filter = build_context_filter(self._context_key, Fact.context_key)
        if context_filter is not None:
            stmt = stmt.where(context_filter)

        # Apply user-provided filter
        if effective_filter:
            clause = effective_filter.build()
            if clause is not None:
                stmt = stmt.where(clause)

        return stmt.scalar_subquery()

    def _hydrate_facts(self, fact_ids: list[int]) -> list[Fact]:
        """Fetch facts from DB by ID. Filtering already done at search time."""
        with self._session_factory() as session:
            stmt = select(Fact).where(Fact.id.in_(fact_ids))
            facts = list(session.scalars(stmt).all())
            return cast(list[Fact], detach_all(facts, session))

    def _build_filter(
        self,
        filter: EmbeddingFilter | None,
        fact_type: str | None,
        categories: list[str] | None,
    ) -> EmbeddingFilter | None:
        """Build effective filter by merging filter param with legacy params.

        If both filter and legacy params are provided, they are ANDed together.
        Does not mutate the input filter.
        """
        if not filter and not fact_type and not categories:
            return None

        # No legacy params to merge - return filter as-is
        needs_merge = (fact_type or categories) and filter is not None
        if filter is not None and not needs_merge:
            return filter

        # Create new filter, copying state from input filter if present
        f = EmbeddingFilter()
        if filter is not None:
            if filter._fact_type:
                f.fact_type(filter._fact_type)
            if filter._categories:
                f.categories(*filter._categories)
            for clause in filter._clauses:
                f.where(clause)

        # Merge legacy params (only if not already set)
        if fact_type and f._fact_type is None:
            f.fact_type(fact_type)
        if categories and f._categories is None:
            f.categories(*categories)

        return f

    def _search_and_score(
        self,
        query: list[float],
        model: str,
        fetch_k: int,
        min_similarity: float,
        effective_filter: EmbeddingFilter | None,
    ) -> list[ScoredEntity[Fact]]:
        """Search embedding store with pre-filtered vector search."""
        # Route to correct store based on query dimension
        store = self._get_store(len(query))

        # Build subquery for pre-filtering (context + active + user filter)
        entity_id_subquery = self._build_entity_id_subquery(effective_filter)

        results = store.search(
            query=query,
            entity_type=self.ENTITY_TYPE,
            model=model,
            top_k=fetch_k,
            min_similarity=min_similarity,
            entity_id_subquery=entity_id_subquery,
        )
        if not results:
            return []

        fact_ids = [int(entity_id) for entity_id, _ in results]
        score_map = {int(entity_id): score for entity_id, score in results}
        facts = self._hydrate_facts(fact_ids)

        scored = [ScoredEntity(entity=f, score=score_map[f.id]) for f in facts if f.id in score_map]
        scored.sort(key=lambda x: x.score, reverse=True)
        return scored

    def search_similar(
        self,
        query: list[float],
        model: str,
        *,
        top_k: int = 10,
        min_similarity: float = 0.0,
        filter: EmbeddingFilter | None = None,
        fact_type: str | None = None,
        categories: list[str] | None = None,
    ) -> list[ScoredEntity[Fact]]:
        """Search for facts similar to a query embedding.

        Routes to the correct dimension table based on query vector length.

        Args:
            query: Query embedding vector.
            model: Embedding model name.
            top_k: Maximum number of results to return.
            min_similarity: Minimum similarity threshold.
            filter: EmbeddingFilter for flexible filtering (recommended).
            fact_type: Legacy filter by fact type (use filter instead).
            categories: Legacy filter by categories (use filter instead).

        Returns:
            List of facts with similarity scores, sorted by similarity.

        Raises:
            ValueError: If top_k is less than 1.
        """
        if top_k < 1:
            raise ValueError(f"top_k must be at least 1, got {top_k}")

        effective_filter = self._build_filter(filter, fact_type, categories)
        return self._search_and_score(query, model, top_k, min_similarity, effective_filter)

    def delete_embedding(
        self, fact_id: int, *, dimensions: int | None = None, session: Session | None = None
    ) -> int:
        """
        Delete embeddings when a fact is deleted.

        Args:
            fact_id: The fact ID.
            dimensions: Embedding dimensions to delete from. If None, uses default_dimensions.
                If both are None, falls back to deleting from cached stores only.
            session: Optional session for transaction participation. If provided,
                the deletion participates in the caller's transaction.

        Returns:
            Number of embeddings deleted.
        """
        dims = dimensions if dimensions is not None else self._default_dimensions
        if dims is not None:
            store = self._get_store(dims)
            return store.delete(self.ENTITY_TYPE, str(fact_id), session=session)
        total = 0
        for store in self._stores.values():
            total += store.delete(self.ENTITY_TYPE, str(fact_id), session=session)
        return total

    def has_embedding(self, fact_id: int, model: str, dimensions: int) -> bool:
        """
        Check if a fact has an embedding for a specific model and dimensions.

        Args:
            fact_id: The fact ID.
            model: Embedding model name.
            dimensions: Embedding dimensions (determines which table to query).

        Returns:
            True if embedding exists.
        """
        store = self._get_store(dimensions)
        return store.exists(self.ENTITY_TYPE, str(fact_id), model)

    def list_without_embeddings(
        self,
        model: str,
        dimensions: int,
        *,
        fact_type: str | None = None,
        limit: int = 100,
    ) -> list[Fact]:
        """
        Find facts that need embeddings for a specific model and dimensions.

        Useful for batch embedding generation.

        Args:
            model: Embedding model name.
            dimensions: Embedding dimensions (determines which table to check).
            fact_type: Optional fact type filter.
            limit: Maximum facts to return.

        Returns:
            List of facts without embeddings for the model.
        """
        store = self._get_store(dimensions)
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
            missing_ids = set(store.list_missing(self.ENTITY_TYPE, fact_ids, model))

            result = [f for f in facts if str(f.id) in missing_ids][:limit]
            return cast(list[Fact], detach_all(result, session))

    def count(self, model: str | None = None, dimensions: int | None = None) -> int:
        """
        Count embeddings for atomic facts.

        Args:
            model: Optional model filter.
            dimensions: Embedding dimensions. If None, uses default_dimensions.
                If both are None, sums across cached stores only.

        Returns:
            Total count.
        """
        dims = dimensions if dimensions is not None else self._default_dimensions
        if dims is not None:
            store = self._get_store(dims)
            return store.count(entity_type=self.ENTITY_TYPE, model=model)
        total = 0
        for store in self._stores.values():
            total += store.count(entity_type=self.ENTITY_TYPE, model=model)
        return total

    def delete_orphans(self, dry_run: bool = False) -> int:
        """Delete embeddings for facts that no longer exist.

        Note:
            Not yet implemented for new quantized embedding tables.

        Args:
            dry_run: If True, count orphans but don't delete them.

        Returns:
            Number of orphan embeddings deleted (or found if dry_run).

        Raises:
            NotImplementedError: This method needs reimplementation for new tables.
        """
        raise NotImplementedError(
            "delete_orphans not yet implemented for quantized embedding tables"
        )
