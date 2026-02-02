"""Assertions client for simple facts about the user."""

from dataclasses import dataclass
from typing import cast

from appinfra.db.utils import detach, detach_all
from sqlalchemy import func, select, text

from ....core.exceptions import ValidationError
from ....core.utils import utc_now
from ..models import Fact
from .base import FactClient


@dataclass
class ScoredFact:
    """A fact with a similarity score from semantic search."""

    fact: Fact
    similarity: float


class AssertionsClient(FactClient[None]):
    """
    Client for simple assertions (facts about the user).

    Assertions are facts that don't need a details table - they just use
    the base memv1_facts table with type='assertion'.

    Examples:
    - "User prefers concise explanations"
    - "User is a Python developer"
    - "User lives in NYC"

    Usage:
        assertions = AssertionsClient(session_factory, profile_id=123)

        # Add an assertion
        fact_id = assertions.add(
            content="User prefers Python over JavaScript",
            category="preferences",
        )

        # List by category
        prefs = assertions.list_by_category("preferences")
    """

    fact_type = "assertion"
    details_model = type(None)  # No details model
    details_relationship = ""  # No details relationship

    def add(
        self,
        content: str,
        category: str | None = None,
        source: str = "user",
        confidence: float = 1.0,
    ) -> int:
        """
        Add an assertion (simple fact).

        Args:
            content: The assertion text
            category: Optional category (preferences, background, rules, etc.)
            source: Source of the assertion (user, inferred, conversation, system)
            confidence: Confidence level 0.0-1.0

        Returns:
            Created fact ID

        Raises:
            ValidationError: If inputs are invalid
        """
        if not content or not content.strip():
            raise ValidationError("content cannot be empty")

        if confidence < 0.0 or confidence > 1.0:
            raise ValidationError(f"confidence must be between 0.0 and 1.0, got {confidence}")

        with self._session_factory() as session:
            fact = Fact(
                profile_id=self.profile_id,
                type=self.fact_type,
                content=content.strip(),
                category=category.strip() if category else None,
                source=source,
                confidence=confidence,
                active=True,
            )
            session.add(fact)
            session.flush()
            return fact.id

    def get(self, fact_id: int) -> Fact | None:
        """
        Get an assertion by ID.

        Args:
            fact_id: The fact ID

        Returns:
            Fact if found, None otherwise
        """
        with self._session_factory() as session:
            fact = self._get_fact(session, fact_id)
            if fact is None:
                return None
            return cast(Fact, detach(fact, session))

    def update(
        self,
        fact_id: int,
        content: str | None = None,
        category: str | None = None,
        confidence: float | None = None,
    ) -> bool:
        """
        Update an assertion.

        Args:
            fact_id: The fact ID to update
            content: New content (if provided)
            category: New category (if provided)
            confidence: New confidence (if provided)

        Returns:
            True if updated, False if not found

        Raises:
            ValidationError: If inputs are invalid
        """
        if content is not None and not content.strip():
            raise ValidationError("content cannot be empty")

        if confidence is not None and (confidence < 0.0 or confidence > 1.0):
            raise ValidationError(f"confidence must be between 0.0 and 1.0, got {confidence}")

        with self._session_factory() as session:
            fact = self._get_fact(session, fact_id)
            if fact is None:
                return False

            if content is not None:
                fact.content = content.strip()
            if category is not None:
                fact.category = category.strip() if category else None
            if confidence is not None:
                fact.confidence = confidence

            fact.updated_at = utc_now()
            return True

    def list_by_category(
        self,
        category: str,
        limit: int = 100,
        active_only: bool = True,
    ) -> list[Fact]:
        """
        List assertions in a specific category.

        Args:
            category: Category to filter by
            limit: Maximum records to return
            active_only: Only return active facts

        Returns:
            List of facts
        """
        with self._session_factory() as session:
            stmt = select(Fact).where(
                Fact.profile_id == self.profile_id,
                Fact.type == self.fact_type,
                Fact.category == category,
            )

            if active_only:
                stmt = stmt.where(Fact.active == True)  # noqa: E712

            stmt = stmt.order_by(Fact.created_at.desc()).limit(limit)
            facts = list(session.scalars(stmt).all())
            return cast(list[Fact], detach_all(facts, session))

    def list_by_source(
        self,
        source: str,
        limit: int = 100,
        active_only: bool = True,
    ) -> list[Fact]:
        """
        List assertions from a specific source.

        Args:
            source: Source to filter by (user, inferred, conversation, system)
            limit: Maximum records to return
            active_only: Only return active facts

        Returns:
            List of facts
        """
        with self._session_factory() as session:
            stmt = select(Fact).where(
                Fact.profile_id == self.profile_id,
                Fact.type == self.fact_type,
                Fact.source == source,
            )

            if active_only:
                stmt = stmt.where(Fact.active == True)  # noqa: E712

            stmt = stmt.order_by(Fact.created_at.desc()).limit(limit)
            facts = list(session.scalars(stmt).all())
            return cast(list[Fact], detach_all(facts, session))

    def search(
        self,
        query: str,
        limit: int = 50,
        active_only: bool = True,
    ) -> list[Fact]:
        """
        Search assertions by content.

        Args:
            query: Text to search for
            limit: Maximum records to return
            active_only: Only return active facts

        Returns:
            List of matching facts
        """
        with self._session_factory() as session:
            stmt = select(Fact).where(
                Fact.profile_id == self.profile_id,
                Fact.type == self.fact_type,
                Fact.content.ilike(f"%{query}%"),
            )

            if active_only:
                stmt = stmt.where(Fact.active == True)  # noqa: E712

            stmt = stmt.order_by(Fact.created_at.desc()).limit(limit)
            facts = list(session.scalars(stmt).all())
            return cast(list[Fact], detach_all(facts, session))

    def get_categories(self) -> list[str]:
        """
        Get list of unique categories.

        Returns:
            List of distinct category names (excluding None)
        """
        with self._session_factory() as session:
            stmt = (
                select(Fact.category)
                .where(
                    Fact.profile_id == self.profile_id,
                    Fact.type == self.fact_type,
                    Fact.category.isnot(None),
                )
                .distinct()
                .order_by(Fact.category)
            )
            return [c for c in session.scalars(stmt).all() if c is not None]

    def list_active(
        self,
        category: str | None = None,
        min_confidence: float = 0.0,
        limit: int = 100,
    ) -> list[Fact]:
        """
        List active assertions with optional filters.

        Args:
            category: Optional category to filter by
            min_confidence: Minimum confidence threshold (default: 0.0)
            limit: Maximum records to return

        Returns:
            List of active facts matching the filters
        """
        with self._session_factory() as session:
            stmt = select(Fact).where(
                Fact.profile_id == self.profile_id,
                Fact.type == self.fact_type,
                Fact.active == True,  # noqa: E712
                Fact.confidence >= min_confidence,
            )

            if category is not None:
                stmt = stmt.where(Fact.category == category)

            stmt = stmt.order_by(Fact.created_at.desc()).limit(limit)
            facts = list(session.scalars(stmt).all())
            return cast(list[Fact], detach_all(facts, session))

    def count_by_category(self) -> dict[str | None, int]:
        """
        Count assertions by category.

        Returns:
            Dict mapping category name (or None) to count
        """
        with self._session_factory() as session:
            stmt = (
                select(Fact.category, func.count(Fact.id))
                .where(
                    Fact.profile_id == self.profile_id,
                    Fact.type == self.fact_type,
                    Fact.active == True,  # noqa: E712
                )
                .group_by(Fact.category)
            )
            results = session.execute(stmt).all()
            return {cat: count for cat, count in results}

    def set_embedding(
        self,
        fact_id: int,
        embedding: list[float],
        model_name: str,
    ) -> None:
        """
        Set or update the embedding for a fact.

        Args:
            fact_id: The fact ID
            embedding: The embedding vector
            model_name: Name of the embedding model

        Raises:
            ValidationError: If fact doesn't exist or doesn't belong to this profile
        """
        from ..models import FactEmbedding

        with self._session_factory() as session:
            fact = self._get_fact(session, fact_id)
            if fact is None:
                raise ValidationError(f"Fact {fact_id} not found")

            # Check for existing embedding
            existing = session.scalars(
                select(FactEmbedding).where(
                    FactEmbedding.fact_id == fact_id,
                    FactEmbedding.model_name == model_name,
                )
            ).first()

            if existing:
                existing.embedding = embedding
                existing.dimensions = len(embedding)
            else:
                emb = FactEmbedding(
                    fact_id=fact_id,
                    model_name=model_name,
                    dimensions=len(embedding),
                    embedding=embedding,
                )
                session.add(emb)

    def list_without_embeddings(
        self,
        model_name: str,
        limit: int = 100,
    ) -> list[Fact]:
        """
        List active facts that don't have embeddings for a specific model.

        Args:
            model_name: Name of the embedding model
            limit: Maximum records to return

        Returns:
            List of facts without embeddings for the given model
        """
        from ..models import FactEmbedding

        with self._session_factory() as session:
            # Subquery for facts with embeddings for this model
            has_embedding = (
                select(FactEmbedding.fact_id)
                .where(FactEmbedding.model_name == model_name)
                .scalar_subquery()
            )

            stmt = (
                select(Fact)
                .where(
                    Fact.profile_id == self.profile_id,
                    Fact.type == self.fact_type,
                    Fact.active == True,  # noqa: E712
                    ~Fact.id.in_(has_embedding),
                )
                .order_by(Fact.created_at.asc())
                .limit(limit)
            )
            facts = list(session.scalars(stmt).all())
            return cast(list[Fact], detach_all(facts, session))

    def _row_to_scored_fact(self, row) -> ScoredFact:
        """Convert a similarity query row to a ScoredFact."""
        fact = Fact(
            id=row.id,
            profile_id=row.profile_id,
            type=row.type,
            content=row.content,
            category=row.category,
            source=row.source,
            confidence=row.confidence,
            active=row.active,
            created_at=row.created_at,
            updated_at=row.updated_at,
        )
        return ScoredFact(fact=fact, similarity=row.similarity)

    def search_similar(
        self,
        embedding: list[float],
        model_name: str,
        top_k: int = 10,
        min_similarity: float = 0.3,
        categories: list[str] | None = None,
        active_only: bool = True,
    ) -> list[ScoredFact]:
        """Search for facts similar to embedding using cosine similarity."""
        with self._session_factory() as session:
            query = self._get_similarity_query(active_only, categories is not None)
            params = {
                "embedding": str(embedding),
                "profile_id": self.profile_id,
                "fact_type": self.fact_type,
                "model_name": model_name,
                "min_similarity": min_similarity,
                "top_k": top_k,
            }
            if categories is not None:
                params["categories"] = categories
            rows = session.execute(query, params).fetchall()
            return [self._row_to_scored_fact(row) for row in rows]

    def _get_similarity_query(self, active_only: bool, has_categories: bool):
        """Build the similarity search SQL query."""
        where_clauses = [
            "f.profile_id = :profile_id",
            "f.type = :fact_type",
            "e.model_name = :model_name",
        ]
        if active_only:
            where_clauses.append("f.active = true")
        if has_categories:
            where_clauses.append("f.category = ANY(:categories)")

        where_sql = " AND ".join(where_clauses)

        # Note: Using CAST() instead of ::vector to avoid conflict with
        # SQLAlchemy's :param syntax which treats ::vector as part of param name
        return text(f"""
            SELECT
                f.id, f.profile_id, f.type, f.content, f.category,
                f.source, f.confidence, f.active, f.created_at, f.updated_at,
                1 - (e.embedding <=> CAST(:embedding AS vector)) as similarity
            FROM memv1_facts f
            JOIN memv1_fact_embeddings e ON e.fact_id = f.id
            WHERE {where_sql}
            AND 1 - (e.embedding <=> CAST(:embedding AS vector)) >= :min_similarity
            ORDER BY e.embedding <=> CAST(:embedding AS vector)
            LIMIT :top_k
        """)
