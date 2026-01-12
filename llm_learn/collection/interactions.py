"""Interactions collection client."""

from datetime import datetime
from typing import Literal, cast

from appinfra.db.utils import detach_all
from sqlalchemy import select

from ..core.exceptions import ValidationError
from ..core.models import Interaction
from .base import ProfileScopedClient

InteractionType = Literal["view", "click", "read", "scroll", "dismiss"]


class InteractionsClient(ProfileScopedClient[Interaction]):
    """
    Client for recording implicit interaction signals scoped to a profile.

    Interactions capture how users engage with content without
    explicit feedback - views, clicks, reading time, scroll depth.

    Usage:
        interactions = InteractionsClient(session_factory, profile_id=123)
        interactions.record(
            content_id=123,
            interaction_type="read",
            duration_ms=45000,
            scroll_depth=0.85,
        )
    """

    model = Interaction

    def record(
        self,
        content_id: int,
        interaction_type: InteractionType,
        duration_ms: int | None = None,
        scroll_depth: float | None = None,
        context: dict | None = None,
    ) -> int:
        """
        Record an interaction.

        Args:
            content_id: Database content ID
            interaction_type: Type of interaction (view, click, read, scroll, dismiss)
            duration_ms: Time spent in milliseconds (for reads)
            scroll_depth: How far scrolled (0.0-1.0)
            context: Additional context

        Returns:
            Created interaction ID

        Raises:
            ValidationError: If inputs are invalid
        """
        # Validate interaction type
        valid_types = ("view", "click", "read", "scroll", "dismiss")
        if interaction_type not in valid_types:
            raise ValidationError(
                f"Invalid interaction type: {interaction_type}. Must be one of {valid_types}"
            )

        # Validate scroll depth
        if scroll_depth is not None and (scroll_depth < 0.0 or scroll_depth > 1.0):
            raise ValidationError(f"Scroll depth must be between 0.0 and 1.0, got {scroll_depth}")

        # Validate duration
        if duration_ms is not None and duration_ms < 0:
            raise ValidationError(f"Duration must be non-negative, got {duration_ms}")

        with self._session_factory() as session:
            interaction = Interaction(
                profile_id=self.profile_id,
                content_id=content_id,
                interaction_type=interaction_type,
                duration_ms=duration_ms,
                scroll_depth=scroll_depth,
                context=context,
            )
            session.add(interaction)
            session.flush()

            return interaction.id

    def list_by_type(
        self,
        interaction_type: InteractionType,
        limit: int = 100,
        since: datetime | None = None,
    ) -> list[Interaction]:
        """
        List interactions by type for this profile.

        Args:
            interaction_type: Type to filter by
            limit: Maximum records to return
            since: Only return interactions after this time

        Returns:
            List of interactions
        """
        with self._session_factory() as session:
            stmt = select(Interaction).where(
                Interaction.profile_id == self.profile_id,
                Interaction.interaction_type == interaction_type,
            )

            if since:
                stmt = stmt.where(Interaction.created_at >= since)

            stmt = stmt.order_by(Interaction.created_at.desc()).limit(limit)
            objects = list(session.scalars(stmt).all())
            return cast(list[Interaction], detach_all(objects, session))

    def list_for_content(
        self,
        content_id: int,
        limit: int = 100,
    ) -> list[Interaction]:
        """
        List all interactions for a specific content item within this profile.

        Args:
            content_id: Content ID to filter by
            limit: Maximum records to return

        Returns:
            List of interactions
        """
        with self._session_factory() as session:
            stmt = (
                select(Interaction)
                .where(
                    Interaction.profile_id == self.profile_id,
                    Interaction.content_id == content_id,
                )
                .order_by(Interaction.created_at.desc())
                .limit(limit)
            )
            objects = list(session.scalars(stmt).all())
            return cast(list[Interaction], detach_all(objects, session))

    def count_by_type(self) -> dict[str, int]:
        """
        Count interactions by type for this profile.

        Returns:
            Dict mapping interaction type to count
        """
        with self._session_factory() as session:
            counts = {}
            for itype in ("view", "click", "read", "scroll", "dismiss"):
                stmt = select(Interaction).where(
                    Interaction.profile_id == self.profile_id,
                    Interaction.interaction_type == itype,
                )
                counts[itype] = len(list(session.scalars(stmt).all()))
            return counts

    def get_engagement_stats(self, content_id: int) -> dict:
        """
        Get engagement statistics for a content item within this profile.

        Args:
            content_id: Content ID

        Returns:
            Dict with view_count, click_count, avg_duration_ms, avg_scroll_depth
        """
        interactions = self.list_for_content(content_id, limit=1000)

        if not interactions:
            return {
                "view_count": 0,
                "click_count": 0,
                "read_count": 0,
                "avg_duration_ms": None,
                "avg_scroll_depth": None,
            }

        view_count = sum(1 for i in interactions if i.interaction_type == "view")
        click_count = sum(1 for i in interactions if i.interaction_type == "click")
        read_count = sum(1 for i in interactions if i.interaction_type == "read")

        durations = [i.duration_ms for i in interactions if i.duration_ms is not None]
        scroll_depths = [i.scroll_depth for i in interactions if i.scroll_depth is not None]

        return {
            "view_count": view_count,
            "click_count": click_count,
            "read_count": read_count,
            "avg_duration_ms": sum(durations) / len(durations) if durations else None,
            "avg_scroll_depth": sum(scroll_depths) / len(scroll_depths) if scroll_depths else None,
        }
