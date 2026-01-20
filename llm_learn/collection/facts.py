"""Facts collection client for context injection."""

import math
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal, cast

from appinfra.db.utils import detach, detach_all
from sqlalchemy import select, text
from sqlalchemy.sql.elements import TextClause

from ..core.exceptions import NotFoundError, ValidationError
from ..core.models import Fact, FactEmbedding
from .base import ProfileScopedClient

FactCategory = Literal["preferences", "background", "rules", "context"]
FactSource = Literal["user", "inferred", "conversation", "system"]


@dataclass
class ScoredFact:
    """Fact with similarity score from vector search."""

    fact: Fact
    similarity: float


def _row_to_fact(row: Any) -> Fact:
    """Convert a database row (from raw SQL) to a Fact object."""
    return Fact(
        id=row.id,
        profile_id=row.profile_id,
        content=row.content,
        category=row.category,
        source=row.source,
        confidence=row.confidence,
        active=row.active,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _validate_embedding(embedding: list[float]) -> None:
    """Validate embedding vector contains valid finite numeric values."""
    if not embedding:
        raise ValidationError("Embedding cannot be empty")
    for i, val in enumerate(embedding):
        if not isinstance(val, (int, float)):
            raise ValidationError(f"Embedding[{i}] must be numeric, got {type(val).__name__}")
        if not math.isfinite(val):
            raise ValidationError(f"Embedding[{i}] must be finite, got {val}")


# Similarity search queries - separate constants to avoid f-string SQL construction
# Variants: active_only (true/false) × categories (with/without)
_SIMILARITY_QUERY_ACTIVE = text("""
    SELECT f.id, f.content, f.category, f.source, f.confidence, f.active,
           f.created_at, f.updated_at, f.profile_id,
           1 - (e.embedding <=> CAST(:embedding AS vector)) as similarity
    FROM facts f
    JOIN fact_embeddings e ON e.fact_id = f.id
    WHERE f.profile_id = :profile_id
      AND e.model_name = :model_name
      AND f.active = true
      AND 1 - (e.embedding <=> CAST(:embedding AS vector)) >= :min_similarity
    ORDER BY e.embedding <=> CAST(:embedding AS vector)
    LIMIT :top_k
""")

_SIMILARITY_QUERY_ACTIVE_WITH_CATEGORIES = text("""
    SELECT f.id, f.content, f.category, f.source, f.confidence, f.active,
           f.created_at, f.updated_at, f.profile_id,
           1 - (e.embedding <=> CAST(:embedding AS vector)) as similarity
    FROM facts f
    JOIN fact_embeddings e ON e.fact_id = f.id
    WHERE f.profile_id = :profile_id
      AND e.model_name = :model_name
      AND f.active = true
      AND f.category = ANY(:categories)
      AND 1 - (e.embedding <=> CAST(:embedding AS vector)) >= :min_similarity
    ORDER BY e.embedding <=> CAST(:embedding AS vector)
    LIMIT :top_k
""")

_SIMILARITY_QUERY_ALL = text("""
    SELECT f.id, f.content, f.category, f.source, f.confidence, f.active,
           f.created_at, f.updated_at, f.profile_id,
           1 - (e.embedding <=> CAST(:embedding AS vector)) as similarity
    FROM facts f
    JOIN fact_embeddings e ON e.fact_id = f.id
    WHERE f.profile_id = :profile_id
      AND e.model_name = :model_name
      AND 1 - (e.embedding <=> CAST(:embedding AS vector)) >= :min_similarity
    ORDER BY e.embedding <=> CAST(:embedding AS vector)
    LIMIT :top_k
""")

_SIMILARITY_QUERY_ALL_WITH_CATEGORIES = text("""
    SELECT f.id, f.content, f.category, f.source, f.confidence, f.active,
           f.created_at, f.updated_at, f.profile_id,
           1 - (e.embedding <=> CAST(:embedding AS vector)) as similarity
    FROM facts f
    JOIN fact_embeddings e ON e.fact_id = f.id
    WHERE f.profile_id = :profile_id
      AND e.model_name = :model_name
      AND f.category = ANY(:categories)
      AND 1 - (e.embedding <=> CAST(:embedding AS vector)) >= :min_similarity
    ORDER BY e.embedding <=> CAST(:embedding AS vector)
    LIMIT :top_k
""")


def _get_similarity_query(active_only: bool, has_categories: bool) -> TextClause:
    """Get the appropriate similarity search query."""
    if active_only:
        return (
            _SIMILARITY_QUERY_ACTIVE_WITH_CATEGORIES if has_categories else _SIMILARITY_QUERY_ACTIVE
        )
    return _SIMILARITY_QUERY_ALL_WITH_CATEGORIES if has_categories else _SIMILARITY_QUERY_ALL


class FactsClient(ProfileScopedClient[Fact]):
    """
    Client for managing user facts scoped to a profile.

    Facts are pieces of information about the user that get injected
    into the system prompt at query time for context injection.

    All operations are scoped to the profile_id provided at construction.

    Usage:
        facts = FactsClient(session_factory, profile_id=123)
        facts.add("Prefers concise explanations", category="preferences")
        facts.add("Timezone: UTC", category="settings")

        # Get all active facts for context injection
        active_facts = facts.list_active()
    """

    model = Fact

    def add(
        self,
        content: str,
        category: str | None = None,
        source: FactSource = "user",
        confidence: float = 1.0,
    ) -> int:
        """
        Add a fact about the user.

        Args:
            content: The fact content (e.g., "Prefers concise explanations")
            category: Category for organization (preferences, background, rules, context)
            source: How the fact was obtained (user, inferred, conversation, system)
            confidence: Confidence level 0.0-1.0 (default 1.0 for user-provided facts)

        Returns:
            Created fact ID

        Raises:
            ValidationError: If inputs are invalid
        """
        if not content or not content.strip():
            raise ValidationError("Fact content cannot be empty")

        if not 0.0 <= confidence <= 1.0:
            raise ValidationError("Confidence must be between 0.0 and 1.0")

        with self._session_factory() as session:
            fact = Fact(
                profile_id=self.profile_id,
                content=content.strip(),
                category=category,
                source=source,
                confidence=confidence,
            )
            session.add(fact)
            session.flush()

            return fact.id

    def update(
        self,
        fact_id: int,
        content: str | None = None,
        category: str | None = None,
        confidence: float | None = None,
        active: bool | None = None,
    ) -> Fact:
        """
        Update a fact.

        Args:
            fact_id: ID of fact to update
            content: New content (optional)
            category: New category (optional)
            confidence: New confidence (optional)
            active: New active status (optional)

        Returns:
            Updated fact

        Raises:
            NotFoundError: If fact not found or belongs to different profile
            ValidationError: If inputs are invalid
        """
        with self._session_factory() as session:
            fact = session.get(Fact, fact_id)
            if not fact or fact.profile_id != self.profile_id:
                raise NotFoundError(f"Fact {fact_id} not found")

            if content is not None:
                if not content.strip():
                    raise ValidationError("Fact content cannot be empty")
                fact.content = content.strip()

            if category is not None:
                fact.category = category

            if confidence is not None:
                if not 0.0 <= confidence <= 1.0:
                    raise ValidationError("Confidence must be between 0.0 and 1.0")
                fact.confidence = confidence

            if active is not None:
                fact.active = active

            fact.updated_at = datetime.now(UTC)
            session.flush()

            return cast(Fact, detach(fact, session))

    def deactivate(self, fact_id: int) -> bool:
        """
        Deactivate a fact (soft delete).

        Args:
            fact_id: ID of fact to deactivate

        Returns:
            True if deactivated

        Raises:
            NotFoundError: If fact not found or belongs to different profile
        """
        with self._session_factory() as session:
            fact = session.get(Fact, fact_id)
            if not fact or fact.profile_id != self.profile_id:
                raise NotFoundError(f"Fact {fact_id} not found")

            fact.active = False
            fact.updated_at = datetime.now(UTC)
            return True

    def activate(self, fact_id: int) -> bool:
        """
        Reactivate a deactivated fact.

        Args:
            fact_id: ID of fact to activate

        Returns:
            True if activated

        Raises:
            NotFoundError: If fact not found or belongs to different profile
        """
        with self._session_factory() as session:
            fact = session.get(Fact, fact_id)
            if not fact or fact.profile_id != self.profile_id:
                raise NotFoundError(f"Fact {fact_id} not found")

            fact.active = True
            fact.updated_at = datetime.now(UTC)
            return True

    def list_active(
        self,
        category: str | None = None,
        min_confidence: float = 0.0,
        limit: int = 100,
    ) -> list[Fact]:
        """
        List active facts for this profile, optionally filtered by category.

        This is the primary method for retrieving facts for context injection.

        Args:
            category: Optional category filter
            min_confidence: Minimum confidence threshold (default 0.0)
            limit: Maximum records to return

        Returns:
            List of active facts
        """
        with self._session_factory() as session:
            stmt = select(Fact).where(
                Fact.profile_id == self.profile_id,
                Fact.active == True,  # noqa: E712
                Fact.confidence >= min_confidence,
            )

            if category:
                stmt = stmt.where(Fact.category == category)

            stmt = stmt.order_by(Fact.created_at.desc()).limit(limit)
            facts = list(session.scalars(stmt).all())
            return cast(list[Fact], detach_all(facts, session))

    def list_by_category(self, category: str, include_inactive: bool = False) -> list[Fact]:
        """
        List facts by category for this profile.

        Args:
            category: Category to filter by
            include_inactive: Whether to include inactive facts

        Returns:
            List of facts in category
        """
        with self._session_factory() as session:
            stmt = select(Fact).where(
                Fact.profile_id == self.profile_id,
                Fact.category == category,
            )

            if not include_inactive:
                stmt = stmt.where(Fact.active == True)  # noqa: E712

            stmt = stmt.order_by(Fact.created_at.desc())
            facts = list(session.scalars(stmt).all())
            return cast(list[Fact], detach_all(facts, session))

    def list_by_source(self, source: FactSource, active_only: bool = True) -> list[Fact]:
        """
        List facts by source for this profile.

        Args:
            source: Source to filter by (user, inferred, conversation, system)
            active_only: Whether to include only active facts

        Returns:
            List of facts from source
        """
        with self._session_factory() as session:
            stmt = select(Fact).where(
                Fact.profile_id == self.profile_id,
                Fact.source == source,
            )

            if active_only:
                stmt = stmt.where(Fact.active == True)  # noqa: E712

            stmt = stmt.order_by(Fact.created_at.desc())
            facts = list(session.scalars(stmt).all())
            return cast(list[Fact], detach_all(facts, session))

    def count_by_category(self, active_only: bool = True) -> dict[str, int]:
        """
        Count facts by category for this profile.

        Args:
            active_only: Whether to count only active facts

        Returns:
            Dict mapping category to count (None category mapped to "uncategorized")
        """
        with self._session_factory() as session:
            stmt = select(Fact).where(Fact.profile_id == self.profile_id)
            if active_only:
                stmt = stmt.where(Fact.active == True)  # noqa: E712

            facts = list(session.scalars(stmt).all())

            counts: dict[str, int] = {}
            for fact in facts:
                cat = fact.category or "uncategorized"
                counts[cat] = counts.get(cat, 0) + 1

            return counts

    def search(self, query: str, active_only: bool = True, limit: int = 50) -> list[Fact]:
        """
        Search facts by content for this profile (case-insensitive substring match).

        Args:
            query: Search query
            active_only: Whether to search only active facts
            limit: Maximum results

        Returns:
            List of matching facts
        """
        with self._session_factory() as session:
            stmt = select(Fact).where(
                Fact.profile_id == self.profile_id,
                Fact.content.ilike(f"%{query}%"),
            )

            if active_only:
                stmt = stmt.where(Fact.active == True)  # noqa: E712

            stmt = stmt.order_by(Fact.created_at.desc()).limit(limit)
            facts = list(session.scalars(stmt).all())
            return cast(list[Fact], detach_all(facts, session))

    def search_similar(
        self,
        embedding: list[float],
        model_name: str,
        top_k: int = 30,
        min_similarity: float = 0.5,
        active_only: bool = True,
        categories: list[str] | None = None,
    ) -> list[ScoredFact]:
        """
        Search facts by embedding similarity using pgvector.

        Args:
            embedding: Query embedding vector.
            model_name: Embedding model name to search against.
            top_k: Maximum number of results.
            min_similarity: Minimum cosine similarity threshold (0-1).
            active_only: Whether to search only active facts.
            categories: Filter results to these categories. None means all categories.

        Returns:
            List of ScoredFact sorted by similarity (highest first).
        """
        with self._session_factory() as session:
            embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"
            has_categories = categories is not None and len(categories) > 0
            params: dict[str, Any] = {
                "embedding": embedding_str,
                "profile_id": self.profile_id,
                "model_name": model_name,
                "min_similarity": min_similarity,
                "top_k": top_k,
            }
            if has_categories:
                params["categories"] = categories
            result = session.execute(
                _get_similarity_query(active_only, has_categories),
                params,
            )
            return [ScoredFact(fact=_row_to_fact(row), similarity=row.similarity) for row in result]

    def set_embedding(
        self,
        fact_id: int,
        embedding: list[float],
        model_name: str,
    ) -> None:
        """
        Set the embedding for a fact (upsert into fact_embeddings table).

        Args:
            fact_id: ID of fact to update.
            embedding: Embedding vector.
            model_name: Name of the embedding model used.

        Raises:
            NotFoundError: If fact not found or belongs to different profile.
            ValidationError: If embedding is invalid.
        """
        _validate_embedding(embedding)

        with self._session_factory() as session:
            fact = session.get(Fact, fact_id)
            if not fact or fact.profile_id != self.profile_id:
                raise NotFoundError(f"Fact {fact_id} not found")

            # Check if embedding already exists for this model
            existing = session.execute(
                select(FactEmbedding).where(
                    FactEmbedding.fact_id == fact_id,
                    FactEmbedding.model_name == model_name,
                )
            ).scalar_one_or_none()

            if existing:
                # Update existing embedding
                existing.embedding = embedding
                existing.dimensions = len(embedding)
            else:
                # Create new embedding
                fact_embedding = FactEmbedding(
                    fact_id=fact_id,
                    model_name=model_name,
                    dimensions=len(embedding),
                    embedding=embedding,
                )
                session.add(fact_embedding)

    def list_without_embeddings(self, model_name: str, limit: int = 100) -> list[Fact]:
        """
        List facts that don't have embeddings for the specified model.

        Args:
            model_name: Embedding model name to check.
            limit: Maximum results.

        Returns:
            List of facts without embeddings for the specified model.
        """
        with self._session_factory() as session:
            # Find facts that don't have an embedding for this model
            query = text("""
                SELECT f.id, f.profile_id, f.content, f.category, f.source,
                       f.confidence, f.active, f.created_at, f.updated_at
                FROM facts f
                LEFT JOIN fact_embeddings e ON e.fact_id = f.id AND e.model_name = :model_name
                WHERE f.profile_id = :profile_id
                  AND f.active = true
                  AND e.id IS NULL
                ORDER BY f.created_at DESC
                LIMIT :limit
            """)

            result = session.execute(
                query,
                {
                    "profile_id": self.profile_id,
                    "model_name": model_name,
                    "limit": limit,
                },
            )

            return [_row_to_fact(row) for row in result]
