"""Directives collection client."""

from datetime import UTC, datetime
from typing import Literal, cast

from appinfra.db.utils import detach_all
from sqlalchemy import select

from ..core.exceptions import NotFoundError, ValidationError
from ..core.models import Directive
from .base import ProfileScopedClient

DirectiveType = Literal["standing", "one-time", "rule"]
DirectiveStatus = Literal["active", "paused", "completed"]


class DirectivesClient(ProfileScopedClient[Directive]):
    """
    Client for managing user directives scoped to a profile.

    Directives are standing instructions that guide LLM responses.
    They represent user goals, preferences, and rules.

    Usage:
        directives = DirectivesClient(session_factory, profile_id=123)
        directives.record(
            text="Focus more on AI safety research",
            directive_type="standing",
        )
    """

    model = Directive

    def record(
        self,
        text: str,
        directive_type: DirectiveType | None = "standing",
        parsed_rules: dict | None = None,
        expires_at: datetime | None = None,
    ) -> int:
        """
        Record a directive.

        Args:
            text: Directive text in natural language
            directive_type: Type of directive (standing, one-time, rule)
            parsed_rules: Extracted rules (if parsed)
            expires_at: When directive expires

        Returns:
            Created directive ID

        Raises:
            ValidationError: If inputs are invalid
        """
        if not text or not text.strip():
            raise ValidationError("Directive text cannot be empty")

        # Validate directive type
        valid_types = ("standing", "one-time", "rule")
        if directive_type and directive_type not in valid_types:
            raise ValidationError(
                f"Invalid directive type: {directive_type}. Must be one of {valid_types}"
            )

        with self._session_factory() as session:
            directive = Directive(
                profile_id=self.profile_id,
                directive_text=text.strip(),
                directive_type=directive_type,
                parsed_rules=parsed_rules,
                expires_at=expires_at,
            )
            session.add(directive)
            session.flush()

            return directive.id

    def set_status(
        self,
        directive_id: int,
        status: DirectiveStatus,
    ) -> bool:
        """
        Update directive status.

        Args:
            directive_id: ID of directive to update
            status: New status (active, paused, completed)

        Returns:
            True if updated successfully

        Raises:
            ValidationError: If status is invalid
            NotFoundError: If directive not found or belongs to different profile
        """
        valid_statuses = ("active", "paused", "completed")
        if status not in valid_statuses:
            raise ValidationError(f"Invalid status: {status}. Must be one of {valid_statuses}")

        with self._session_factory() as session:
            directive = session.get(Directive, directive_id)
            if not directive or directive.profile_id != self.profile_id:
                raise NotFoundError(f"Directive {directive_id} not found")

            directive.status = status
            return True

    def update_parsed_rules(
        self,
        directive_id: int,
        parsed_rules: dict,
    ) -> bool:
        """
        Update parsed rules for a directive.

        Args:
            directive_id: ID of directive to update
            parsed_rules: New parsed rules

        Returns:
            True if updated successfully

        Raises:
            NotFoundError: If directive not found or belongs to different profile
        """
        with self._session_factory() as session:
            directive = session.get(Directive, directive_id)
            if not directive or directive.profile_id != self.profile_id:
                raise NotFoundError(f"Directive {directive_id} not found")

            directive.parsed_rules = parsed_rules
            return True

    def list_active(self, limit: int = 100) -> list[Directive]:
        """
        List active directives for this profile.

        Args:
            limit: Maximum records to return

        Returns:
            List of active directives
        """
        with self._session_factory() as session:
            now = datetime.now(UTC)
            stmt = (
                select(Directive)
                .where(
                    Directive.profile_id == self.profile_id,
                    Directive.status == "active",
                )
                .order_by(Directive.created_at.desc())
                .limit(limit)
            )
            directives = list(session.scalars(stmt).all())

            # Filter out expired directives and detach remaining
            result = [d for d in directives if d.expires_at is None or d.expires_at > now]
            return cast(list[Directive], detach_all(result, session))

    def list_by_type(
        self,
        directive_type: DirectiveType,
        status: DirectiveStatus | None = None,
        limit: int = 100,
    ) -> list[Directive]:
        """
        List directives by type for this profile.

        Args:
            directive_type: Type to filter by
            status: Optional status filter
            limit: Maximum records to return

        Returns:
            List of directives
        """
        with self._session_factory() as session:
            stmt = select(Directive).where(
                Directive.profile_id == self.profile_id,
                Directive.directive_type == directive_type,
            )

            if status:
                stmt = stmt.where(Directive.status == status)

            stmt = stmt.order_by(Directive.created_at.desc()).limit(limit)
            objects = list(session.scalars(stmt).all())
            return cast(list[Directive], detach_all(objects, session))

    def list_expired(self, as_of: datetime | None = None) -> list[Directive]:
        """
        List expired directives that are still active for this profile.

        Args:
            as_of: Time to check against (default: now)

        Returns:
            List of expired but still active directives
        """
        if as_of is None:
            as_of = datetime.now(UTC)

        with self._session_factory() as session:
            stmt = (
                select(Directive)
                .where(
                    Directive.profile_id == self.profile_id,
                    Directive.status == "active",
                    Directive.expires_at.isnot(None),
                    Directive.expires_at <= as_of,
                )
                .order_by(Directive.expires_at.asc())
            )
            objects = list(session.scalars(stmt).all())
            return cast(list[Directive], detach_all(objects, session))

    def count_by_status(self) -> dict[str, int]:
        """
        Count directives by status for this profile.

        Returns:
            Dict mapping status to count
        """
        with self._session_factory() as session:
            counts = {}
            for status in ("active", "paused", "completed"):
                stmt = select(Directive).where(
                    Directive.profile_id == self.profile_id,
                    Directive.status == status,
                )
                counts[status] = len(list(session.scalars(stmt).all()))
            return counts
