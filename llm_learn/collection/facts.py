"""Facts collection client for context injection."""

from datetime import UTC, datetime
from typing import Literal, cast

from appinfra.db.utils import detach, detach_all
from sqlalchemy import select

from ..core.exceptions import NotFoundError, ValidationError
from ..core.models import Fact
from .base import ProfileScopedClient

FactCategory = Literal["preferences", "background", "rules", "context"]
FactSource = Literal["user", "inferred", "conversation", "system"]


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
