"""Interactions client for implicit behavioral signals."""

import uuid
from typing import Literal, cast

from appinfra.db.utils import detach, detach_all
from sqlalchemy import func, select

from llm_kelt.core.exceptions import ValidationError

from ..models import Fact, InteractionDetails
from .base import FactClient

InteractionType = Literal["view", "click", "read", "scroll", "dismiss"]


class InteractionsClient(FactClient[InteractionDetails]):
    """
    Client for recording implicit user interactions.

    Interactions capture behavioral signals without explicit feedback:
    - view: User saw content
    - click: User clicked/tapped
    - read: User read content (time-based)
    - scroll: User scrolled through content
    - dismiss: User dismissed without engagement

    Usage:
        interactions = InteractionsClient(session_factory, context_key="my-agent")

        # Record a read interaction
        fact_id = interactions.record(
            interaction_type="read",
            content_id=456,
            duration_ms=45000,
            scroll_depth=0.8,
        )
    """

    fact_type = "interaction"
    details_model = InteractionDetails
    details_relationship = "interaction_details"

    def _validate_interaction_inputs(
        self,
        interaction_type: InteractionType,
        scroll_depth: float | None,
        duration_ms: int | None,
    ) -> None:
        """Validate interaction record inputs."""
        valid_types = ("view", "click", "read", "scroll", "dismiss")
        if interaction_type not in valid_types:
            raise ValidationError(f"Invalid interaction_type: {interaction_type}")
        if scroll_depth is not None and (scroll_depth < 0.0 or scroll_depth > 1.0):
            raise ValidationError(f"scroll_depth must be between 0.0 and 1.0, got {scroll_depth}")
        if duration_ms is not None and duration_ms < 0:
            raise ValidationError(f"duration_ms must be non-negative, got {duration_ms}")

    def record(
        self,
        interaction_type: InteractionType,
        content_id: int | None = None,
        duration_ms: int | None = None,
        scroll_depth: float | None = None,
        context: dict | None = None,
        category: str | None = None,
    ) -> int:
        """Record an interaction."""
        self._validate_interaction_inputs(interaction_type, scroll_depth, duration_ms)

        with self._session_factory() as session:
            content_desc = f" on content {content_id}" if content_id else ""
            content_text = f"{interaction_type} interaction{content_desc}"
            # Include UUID to ensure unique content_hash for each interaction event
            unique_content = f"{content_text}|{uuid.uuid4()}"
            fact = Fact(
                context_key=self.context_key,
                type=self.fact_type,
                content=content_text,
                content_hash=self._compute_content_hash(unique_content),
                category=category,
                source="observed",
                confidence=1.0,
                active=True,
            )
            session.add(fact)
            session.flush()

            details = InteractionDetails(
                fact_id=fact.id,
                content_id=content_id,
                interaction_type=interaction_type,
                duration_ms=duration_ms,
                scroll_depth=scroll_depth,
                context=context,
            )
            session.add(details)
            return fact.id

    def list_by_type(
        self,
        interaction_type: InteractionType,
        limit: int = 100,
    ) -> list[Fact]:
        """List interactions by type."""
        with self._session_factory() as session:
            stmt = (
                select(Fact)
                .join(InteractionDetails)
                .where(
                    Fact.context_key == self.context_key,
                    Fact.type == self.fact_type,
                    InteractionDetails.interaction_type == interaction_type,
                )
                .order_by(Fact.created_at.desc())
                .limit(limit)
            )

            facts = list(session.scalars(stmt).all())
            for fact in facts:
                details = fact.interaction_details
                if details is not None:
                    detach(details, session)
            return cast(list[Fact], detach_all(facts, session))

    def list_by_content(
        self,
        content_id: int,
        limit: int = 100,
    ) -> list[Fact]:
        """List all interactions for a specific content item."""
        with self._session_factory() as session:
            stmt = (
                select(Fact)
                .join(InteractionDetails)
                .where(
                    Fact.context_key == self.context_key,
                    Fact.type == self.fact_type,
                    InteractionDetails.content_id == content_id,
                )
                .order_by(Fact.created_at.desc())
                .limit(limit)
            )

            facts = list(session.scalars(stmt).all())
            for fact in facts:
                details = fact.interaction_details
                if details is not None:
                    detach(details, session)
            return cast(list[Fact], detach_all(facts, session))

    def count_by_type(self) -> dict[str, int]:
        """Count interactions by type."""
        with self._session_factory() as session:
            counts = {}
            for itype in ("view", "click", "read", "scroll", "dismiss"):
                stmt = (
                    select(func.count())
                    .select_from(InteractionDetails)
                    .join(Fact)
                    .where(
                        Fact.context_key == self.context_key,
                        Fact.type == self.fact_type,
                        InteractionDetails.interaction_type == itype,
                    )
                )
                counts[itype] = session.scalar(stmt) or 0
            return counts

    def get_engagement_stats(self, content_id: int) -> dict[str, int | float | None]:
        """Get engagement statistics for a content item."""
        with self._session_factory() as session:
            stmt = (
                select(
                    func.count().label("count"),
                    func.sum(InteractionDetails.duration_ms).label("total_duration"),
                    func.avg(InteractionDetails.scroll_depth).label("avg_scroll"),
                )
                .select_from(InteractionDetails)
                .join(Fact)
                .where(
                    Fact.context_key == self.context_key,
                    Fact.type == self.fact_type,
                    InteractionDetails.content_id == content_id,
                )
            )

            result = session.execute(stmt).one()
            return {
                "interaction_count": result.count or 0,
                "total_duration_ms": result.total_duration or 0,
                "avg_scroll_depth": float(result.avg_scroll)
                if result.avg_scroll is not None
                else None,
            }
