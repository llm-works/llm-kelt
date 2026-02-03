"""Directives client for standing user instructions."""

from datetime import datetime
from typing import Literal, cast

from appinfra.db.utils import detach, detach_all
from sqlalchemy import select

from ....core.exceptions import ValidationError
from ....core.utils import utc_now
from ..models import DirectiveDetails, Fact
from .base import FactClient

DirectiveType = Literal["standing", "one-time", "rule"]
StatusType = Literal["active", "paused", "completed"]


class DirectivesClient(FactClient[DirectiveDetails]):
    """
    Client for managing standing user directives/instructions.

    Directives are standing instructions that should influence behavior:
    - standing: Always apply (e.g., "Always use type hints")
    - one-time: Apply once then complete
    - rule: Conditional rules (e.g., "If X, then Y")

    Usage:
        directives = DirectivesClient(session_factory, profile_id=123)

        # Add a directive
        fact_id = directives.record(
            text="Always use Python type hints in code",
            directive_type="standing",
        )

        # Pause/resume
        directives.set_status(fact_id, "paused")
        directives.set_status(fact_id, "active")
    """

    fact_type = "directive"
    details_model = DirectiveDetails
    details_relationship = "directive_details"

    def _validate_directive(self, text: str, directive_type: DirectiveType) -> None:
        """Validate directive inputs."""
        if not text or not text.strip():
            raise ValidationError("text cannot be empty")
        valid_types = ("standing", "one-time", "rule")
        if directive_type not in valid_types:
            raise ValidationError(f"Invalid directive_type: {directive_type}")

    def record(
        self,
        text: str,
        directive_type: DirectiveType = "standing",
        parsed_rules: dict | None = None,
        expires_at: datetime | None = None,
        category: str | None = None,
    ) -> int:
        """Record a directive. See class docstring for full usage."""
        self._validate_directive(text, directive_type)

        with self._session_factory() as session:
            fact = Fact(
                profile_id=self.profile_id,
                type=self.fact_type,
                content=text.strip(),
                category=category,
                source="user",
                confidence=1.0,
                active=True,
            )
            session.add(fact)
            session.flush()
            details = DirectiveDetails(
                fact_id=fact.id,
                directive_type=directive_type,
                parsed_rules=parsed_rules,
                status="active",
                expires_at=expires_at,
            )
            session.add(details)
            return fact.id

    def set_status(self, fact_id: int, status: StatusType) -> bool:
        """
        Set directive status.

        Args:
            fact_id: The fact ID
            status: New status (active, paused, completed)

        Returns:
            True if updated, False if not found

        Raises:
            ValidationError: If status is invalid
        """
        valid_statuses = ("active", "paused", "completed")
        if status not in valid_statuses:
            raise ValidationError(f"Invalid status: {status}")

        with self._session_factory() as session:
            fact = self._get_fact(session, fact_id)
            if fact is None:
                return False

            details = fact.directive_details
            if details is None:
                return False

            details.status = status
            fact.active = status == "active"
            fact.updated_at = utc_now()
            return True

    def list_active(
        self,
        directive_type: DirectiveType | None = None,
        limit: int = 100,
    ) -> list[Fact]:
        """
        List active directives.

        Args:
            directive_type: Optional type filter
            limit: Maximum records to return

        Returns:
            List of active directives
        """
        with self._session_factory() as session:
            stmt = (
                select(Fact)
                .join(DirectiveDetails)
                .where(
                    Fact.profile_id == self.profile_id,
                    Fact.type == self.fact_type,
                    DirectiveDetails.status == "active",
                )
            )

            if directive_type:
                stmt = stmt.where(DirectiveDetails.directive_type == directive_type)

            # Filter out expired
            now = utc_now()
            stmt = stmt.where(
                (DirectiveDetails.expires_at.is_(None)) | (DirectiveDetails.expires_at > now)
            )

            stmt = stmt.order_by(Fact.created_at.desc()).limit(limit)

            facts = list(session.scalars(stmt).all())
            for fact in facts:
                details = fact.directive_details
                if details is not None:
                    detach(details, session)
            return cast(list[Fact], detach_all(facts, session))

    def list_by_type(
        self,
        directive_type: DirectiveType,
        include_inactive: bool = False,
        limit: int = 100,
    ) -> list[Fact]:
        """
        List directives by type.

        Args:
            directive_type: Type to filter by
            include_inactive: Include paused/completed directives
            limit: Maximum records to return

        Returns:
            List of directives
        """
        with self._session_factory() as session:
            stmt = (
                select(Fact)
                .join(DirectiveDetails)
                .where(
                    Fact.profile_id == self.profile_id,
                    Fact.type == self.fact_type,
                    DirectiveDetails.directive_type == directive_type,
                )
            )

            if not include_inactive:
                stmt = stmt.where(DirectiveDetails.status == "active")
                # Also filter out expired directives
                now = utc_now()
                stmt = stmt.where(
                    (DirectiveDetails.expires_at.is_(None)) | (DirectiveDetails.expires_at > now)
                )

            stmt = stmt.order_by(Fact.created_at.desc()).limit(limit)

            facts = list(session.scalars(stmt).all())
            for fact in facts:
                details = fact.directive_details
                if details is not None:
                    detach(details, session)
            return cast(list[Fact], detach_all(facts, session))
