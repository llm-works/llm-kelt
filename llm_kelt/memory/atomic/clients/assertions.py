"""Assertions client for simple facts about the user."""

from datetime import UTC, datetime
from typing import cast

from appinfra.db.utils import detach, detach_all
from sqlalchemy import func, select

from llm_kelt.core.errors import ValidationError

from ..models import Fact
from .base import FactClient


class AssertionsClient(FactClient[None]):
    """
    Client for simple assertions (facts about the user).

    Assertions are facts that don't need a details table - they just use
    the base atomic_facts table with type='assertion'.

    Examples:
    - "User prefers concise explanations"
    - "User is a Python developer"
    - "User lives in NYC"

    Usage:
        assertions = AssertionsClient(session_factory, context_key="my-agent")

        # Add an assertion
        fact_id = assertions.add(
            content="User prefers Python over JavaScript",
            category="preferences",
        )

        # List by category
        prefs = assertions.list_by_category("preferences")

    Note:
        Embedding operations are handled separately via Protocol.embeddings.
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
            content: The assertion text.
            category: Optional category (preferences, background, rules, etc.).
            source: Source of the assertion (user, inferred, conversation, system).
            confidence: Confidence level 0.0-1.0.

        Returns:
            Created fact ID.

        Raises:
            ValidationError: If inputs are invalid.
        """
        if not content or not content.strip():
            raise ValidationError("content cannot be empty")

        if not (0.0 <= confidence <= 1.0):
            raise ValidationError(f"confidence must be between 0.0 and 1.0, got {confidence}")

        with self._session_factory() as session:
            content_text = content.strip()
            fact = Fact(
                context_key=self.context_key,
                type=self.fact_type,
                content=content_text,
                content_hash=self._compute_content_hash(content_text),
                category=category.strip() if category else None,
                source=source,
                confidence=confidence,
                active=True,
            )
            session.add(fact)
            session.flush()

            # Auto-embed if embedder configured
            self._auto_embed_fact(fact, session)

            return fact.id

    def get(self, fact_id: int) -> Fact | None:
        """
        Get an assertion by ID.

        Args:
            fact_id: The fact ID.

        Returns:
            Fact if found, None otherwise.
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
            fact_id: The fact ID to update.
            content: New content (if provided).
            category: New category (if provided).
            confidence: New confidence (if provided).

        Returns:
            True if updated, False if not found.

        Raises:
            ValidationError: If inputs are invalid.
        """
        if content is not None and not content.strip():
            raise ValidationError("content cannot be empty")

        if confidence is not None and not (0.0 <= confidence <= 1.0):
            raise ValidationError(f"confidence must be between 0.0 and 1.0, got {confidence}")

        with self._session_factory() as session:
            fact = self._get_fact(session, fact_id)
            if fact is None:
                return False

            if content is not None:
                content_text = content.strip()
                fact.content = content_text
                fact.content_hash = self._compute_content_hash(content_text)
            if category is not None:
                fact.category = category.strip() if category else None
            if confidence is not None:
                fact.confidence = confidence

            fact.updated_at = datetime.now(UTC)
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
            category: Category to filter by.
            limit: Maximum records to return.
            active_only: Only return active facts.

        Returns:
            List of facts.
        """
        with self._session_factory() as session:
            stmt = select(Fact).where(
                Fact.type == self.fact_type,
                Fact.category == category,
            )

            context_filter = self._build_context_filter(Fact.context_key)
            if context_filter is not None:
                stmt = stmt.where(context_filter)

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
            source: Source to filter by (user, inferred, conversation, system).
            limit: Maximum records to return.
            active_only: Only return active facts.

        Returns:
            List of facts.
        """
        with self._session_factory() as session:
            stmt = select(Fact).where(
                Fact.type == self.fact_type,
                Fact.source == source,
            )

            context_filter = self._build_context_filter(Fact.context_key)
            if context_filter is not None:
                stmt = stmt.where(context_filter)

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
        Search assertions by content (text search).

        Args:
            query: Text to search for.
            limit: Maximum records to return.
            active_only: Only return active facts.

        Returns:
            List of matching facts.

        Note:
            For semantic search, use Protocol.embeddings.search_similar().
        """
        with self._session_factory() as session:
            stmt = select(Fact).where(
                Fact.type == self.fact_type,
                Fact.content.ilike(f"%{query}%"),
            )

            context_filter = self._build_context_filter(Fact.context_key)
            if context_filter is not None:
                stmt = stmt.where(context_filter)

            if active_only:
                stmt = stmt.where(Fact.active == True)  # noqa: E712

            stmt = stmt.order_by(Fact.created_at.desc()).limit(limit)
            facts = list(session.scalars(stmt).all())
            return cast(list[Fact], detach_all(facts, session))

    def get_categories(self) -> list[str]:
        """
        Get list of unique categories.

        Returns:
            List of distinct category names (excluding None).
        """
        with self._session_factory() as session:
            stmt = select(Fact.category).where(
                Fact.type == self.fact_type,
                Fact.category.isnot(None),
            )

            context_filter = self._build_context_filter(Fact.context_key)
            if context_filter is not None:
                stmt = stmt.where(context_filter)

            stmt = stmt.distinct().order_by(Fact.category)
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
            category: Optional category to filter by.
            min_confidence: Minimum confidence threshold (default: 0.0).
            limit: Maximum records to return.

        Returns:
            List of active facts matching the filters.
        """
        with self._session_factory() as session:
            stmt = select(Fact).where(
                Fact.type == self.fact_type,
                Fact.active == True,  # noqa: E712
                Fact.confidence >= min_confidence,
            )

            context_filter = self._build_context_filter(Fact.context_key)
            if context_filter is not None:
                stmt = stmt.where(context_filter)

            if category is not None:
                stmt = stmt.where(Fact.category == category)

            stmt = stmt.order_by(Fact.created_at.desc()).limit(limit)
            facts = list(session.scalars(stmt).all())
            return cast(list[Fact], detach_all(facts, session))

    def count_by_category(self) -> dict[str | None, int]:
        """
        Count assertions by category.

        Returns:
            Dict mapping category name (or None) to count.
        """
        with self._session_factory() as session:
            stmt = select(Fact.category, func.count(Fact.id)).where(
                Fact.type == self.fact_type,
                Fact.active == True,  # noqa: E712
            )

            context_filter = self._build_context_filter(Fact.context_key)
            if context_filter is not None:
                stmt = stmt.where(context_filter)

            stmt = stmt.group_by(Fact.category)
            results = session.execute(stmt).all()
            return {cat: count for cat, count in results}
