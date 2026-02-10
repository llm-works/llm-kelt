"""Base client for atomic memory fact-based storage."""

from collections.abc import Callable
from typing import Any, Generic, TypeVar, cast

from appinfra.db.utils import detach, detach_all
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from llm_learn.core.base import utc_now

from ..models import Fact

T = TypeVar("T")  # Details model type


class FactClient(Generic[T]):
    """
    Base client for atomic memory collections.

    Handles the fact + details pattern where each record consists of:
    - A base fact record (in atomic_facts)
    - A details record (in atomic_*_details)

    Subclasses must set:
    - fact_type: The type string for atomic_facts.type
    - details_model: The SQLAlchemy model for the details table
    - details_relationship: The relationship name on Fact model
    """

    fact_type: str  # e.g., "solution", "prediction"
    details_model: type[T]  # e.g., SolutionDetails
    details_relationship: str  # e.g., "solution_details"

    def __init__(self, session_factory: Callable[[], Any], context_key: str | None) -> None:
        """
        Initialize client scoped to a specific context.

        Args:
            session_factory: Callable that returns a context manager for database sessions.
            context_key: Context key to scope all operations to (None = no filtering).
                Supports SQL LIKE patterns (% and _) for prefix/pattern matching.
                Examples:
                  - "acme:prod:reviewer" - exact match
                  - "acme:prod:%" - all profiles in workspace
                  - "acme:%" - all workspaces in domain
        """
        self._session_factory = session_factory
        self.context_key = context_key

    def _build_context_filter(self, column):
        """
        Build context filter condition with pattern matching support.

        Args:
            column: SQLAlchemy column to filter on.

        Returns:
            SQLAlchemy filter condition, or None if no filtering needed.
        """
        from llm_learn.memory.isolation import build_context_filter

        return build_context_filter(self.context_key, column)

    def _get_fact(self, session: Session, fact_id: int) -> Fact | None:
        """Get fact by ID, verifying context ownership."""
        from sqlalchemy import select

        # Build query with all filters upfront (single query)
        stmt = select(Fact).where(Fact.id == fact_id, Fact.type == self.fact_type)

        # Apply context filter if needed (supports glob patterns)
        context_filter = self._build_context_filter(Fact.context_key)
        if context_filter is not None:
            stmt = stmt.where(context_filter)

        return session.scalar(stmt)

    def get(self, fact_id: int) -> Fact | None:
        """
        Get a fact with its details by ID.

        Args:
            fact_id: The fact ID.

        Returns:
            Fact with details loaded, or None if not found.
        """
        with self._session_factory() as session:
            fact = self._get_fact(session, fact_id)
            if fact is None:
                return None
            # Eagerly load the details relationship (if any)
            if self.details_relationship:
                details = getattr(fact, self.details_relationship)
                if details is not None:
                    detach(details, session)
            # Detach fact from session
            detach(fact, session)
            return fact

    def list(
        self,
        limit: int = 100,
        offset: int = 0,
        descending: bool = True,
        active_only: bool = True,
    ) -> list[Fact]:
        """
        List facts of this type for the context.

        Args:
            limit: Maximum records to return.
            offset: Records to skip.
            descending: Order by created_at descending (newest first).
            active_only: Only return active facts.

        Returns:
            List of facts with details loaded.
        """
        with self._session_factory() as session:
            stmt = select(Fact).where(Fact.type == self.fact_type)

            # Apply context filter (supports pattern matching)
            context_filter = self._build_context_filter(Fact.context_key)
            if context_filter is not None:
                stmt = stmt.where(context_filter)

            if active_only:
                stmt = stmt.where(Fact.active == True)  # noqa: E712

            order = Fact.created_at.desc() if descending else Fact.created_at.asc()
            stmt = stmt.order_by(order).limit(limit).offset(offset)

            facts = list(session.scalars(stmt).all())
            # Load and detach details for each fact
            for fact in facts:
                if self.details_relationship:
                    details = getattr(fact, self.details_relationship)
                    if details is not None:
                        detach(details, session)
            return cast(list[Fact], detach_all(facts, session))

    def count(self, active_only: bool = True) -> int:
        """
        Count facts of this type for the context.

        Args:
            active_only: Only count active facts.

        Returns:
            Count of matching facts.
        """
        with self._session_factory() as session:
            stmt = select(func.count()).select_from(Fact).where(Fact.type == self.fact_type)

            # Apply context filter (supports pattern matching)
            context_filter = self._build_context_filter(Fact.context_key)
            if context_filter is not None:
                stmt = stmt.where(context_filter)

            if active_only:
                stmt = stmt.where(Fact.active == True)  # noqa: E712

            return session.scalar(stmt) or 0

    def delete(self, fact_id: int) -> bool:
        """
        Delete a fact and its details (hard delete).

        Args:
            fact_id: The fact ID to delete.

        Returns:
            True if deleted, False if not found.
        """
        with self._session_factory() as session:
            fact = self._get_fact(session, fact_id)
            if fact is None:
                return False
            session.delete(fact)  # CASCADE deletes details
            return True

    def deactivate(self, fact_id: int) -> bool:
        """
        Deactivate a fact (soft delete).

        Args:
            fact_id: The fact ID to deactivate.

        Returns:
            True if deactivated, False if not found.
        """
        with self._session_factory() as session:
            fact = self._get_fact(session, fact_id)
            if fact is None:
                return False
            fact.active = False
            fact.updated_at = utc_now()
            return True

    def activate(self, fact_id: int) -> bool:
        """
        Activate a fact (undo soft delete).

        Args:
            fact_id: The fact ID to activate.

        Returns:
            True if activated, False if not found.
        """
        with self._session_factory() as session:
            fact = self._get_fact(session, fact_id)
            if fact is None:
                return False
            fact.active = True
            fact.updated_at = utc_now()
            return True

    def exists(self, fact_id: int) -> bool:
        """
        Check if a fact exists and belongs to this context.

        Args:
            fact_id: The fact ID.

        Returns:
            True if exists and belongs to context.
        """
        with self._session_factory() as session:
            stmt = (
                select(func.count())
                .select_from(Fact)
                .where(
                    Fact.id == fact_id,
                    Fact.type == self.fact_type,
                )
            )

            # Apply context filter (supports pattern matching)
            context_filter = self._build_context_filter(Fact.context_key)
            if context_filter is not None:
                stmt = stmt.where(context_filter)
            return (session.scalar(stmt) or 0) > 0
