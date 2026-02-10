"""Feedback client for explicit user signals on content."""

import math
import uuid
from typing import Literal, cast

from appinfra.db.utils import detach, detach_all
from sqlalchemy import func, select

from llm_learn.core.exceptions import ValidationError

from ..models import Fact, FeedbackDetails
from .base import FactClient

SignalType = Literal["positive", "negative", "dismiss"]


class FeedbackClient(FactClient[FeedbackDetails]):
    """
    Client for recording explicit user feedback on content.

    Feedback represents user signals (positive, negative, dismiss) on
    content items, with optional strength and tags.

    Usage:
        feedback = FeedbackClient(session_factory, context_key="my-agent")

        # Record feedback
        fact_id = feedback.record(
            signal="positive",
            content_id=456,
            strength=0.9,
            comment="Great article!",
            tags=["informative", "well-written"],
        )

        # List by signal
        positive = feedback.list_by_signal("positive")
    """

    fact_type = "feedback"
    details_model = FeedbackDetails
    details_relationship = "feedback_details"

    def _validate_feedback_inputs(self, signal: SignalType, strength: float) -> None:
        """Validate feedback record inputs."""
        valid_signals = ("positive", "negative", "dismiss")
        if signal not in valid_signals:
            raise ValidationError(f"Invalid signal: {signal}. Must be one of {valid_signals}")
        if not math.isfinite(strength) or strength < 0.0 or strength > 1.0:
            raise ValidationError(f"strength must be between 0.0 and 1.0, got {strength}")

    def _create_feedback_fact(
        self, signal: SignalType, content_id: int | None, strength: float, category: str | None
    ) -> Fact:
        """Create a Fact for feedback."""
        content_desc = f" on content {content_id}" if content_id else ""
        content_text = f"{signal} feedback{content_desc}"
        # Include UUID to ensure unique content_hash for each feedback event
        unique_content = f"{content_text}|{uuid.uuid4()}"
        return Fact(
            context_key=self.context_key,
            type=self.fact_type,
            content=content_text,
            content_hash=self._compute_content_hash(unique_content),
            category=category,
            source="user",
            confidence=strength,
            active=True,
        )

    def record(
        self,
        signal: SignalType,
        content_id: int | None = None,
        strength: float = 1.0,
        tags: list[str] | None = None,
        comment: str | None = None,
        context: dict | None = None,
        category: str | None = None,
    ) -> int:
        """Record user feedback."""
        self._validate_feedback_inputs(signal, strength)

        with self._session_factory() as session:
            fact = self._create_feedback_fact(signal, content_id, strength, category)
            session.add(fact)
            session.flush()

            details = FeedbackDetails(
                fact_id=fact.id,
                content_id=content_id,
                signal=signal,
                strength=strength,
                tags=tags,
                comment=comment,
                context=context,
            )
            session.add(details)
            return fact.id

    def list_by_signal(
        self,
        signal: SignalType,
        limit: int = 100,
        active_only: bool = True,
    ) -> list[Fact]:
        """List feedback by signal type."""
        with self._session_factory() as session:
            stmt = (
                select(Fact)
                .join(FeedbackDetails)
                .where(
                    Fact.type == self.fact_type,
                    FeedbackDetails.signal == signal,
                )
            )

            context_filter = self._build_context_filter(Fact.context_key)
            if context_filter is not None:
                stmt = stmt.where(context_filter)

            if active_only:
                stmt = stmt.where(Fact.active == True)  # noqa: E712

            stmt = stmt.order_by(Fact.created_at.desc()).limit(limit)

            facts = list(session.scalars(stmt).all())
            for fact in facts:
                details = fact.feedback_details
                if details is not None:
                    detach(details, session)
            return cast(list[Fact], detach_all(facts, session))

    def list_by_content(
        self,
        content_id: int,
        limit: int = 100,
    ) -> list[Fact]:
        """List all feedback for a specific content item."""
        with self._session_factory() as session:
            stmt = (
                select(Fact)
                .join(FeedbackDetails)
                .where(
                    Fact.type == self.fact_type,
                    FeedbackDetails.content_id == content_id,
                )
            )

            context_filter = self._build_context_filter(Fact.context_key)
            if context_filter is not None:
                stmt = stmt.where(context_filter)

            stmt = stmt.order_by(Fact.created_at.desc()).limit(limit)

            facts = list(session.scalars(stmt).all())
            for fact in facts:
                details = fact.feedback_details
                if details is not None:
                    detach(details, session)
            return cast(list[Fact], detach_all(facts, session))

    def count_by_signal(self) -> dict[str, int]:
        """Count feedback by signal type."""
        with self._session_factory() as session:
            counts = {}
            for signal in ("positive", "negative", "dismiss"):
                stmt = (
                    select(func.count())
                    .select_from(FeedbackDetails)
                    .join(Fact)
                    .where(
                        Fact.type == self.fact_type,
                        FeedbackDetails.signal == signal,
                    )
                )

                context_filter = self._build_context_filter(Fact.context_key)
                if context_filter is not None:
                    stmt = stmt.where(context_filter)

                counts[signal] = session.scalar(stmt) or 0
            return counts
