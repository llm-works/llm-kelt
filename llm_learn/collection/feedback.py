"""Feedback collection client."""

from datetime import datetime
from typing import Literal, cast

from appinfra.db.utils import detach, detach_all
from sqlalchemy import select

from ..core.exceptions import ValidationError
from ..core.models import Content, Feedback
from .base import ProfileScopedClient
from .content import ContentClient

SignalType = Literal["positive", "negative", "dismiss"]


class FeedbackClient(ProfileScopedClient[Feedback]):
    """
    Client for recording explicit feedback signals scoped to a profile.

    Feedback represents user judgment on content quality/relevance.

    Usage:
        feedback = FeedbackClient(session_factory, profile_id=123)
        feedback.record(
            content_id="article_123",
            content_text="Article content...",
            signal="positive",
            strength=0.9,
        )
    """

    model = Feedback

    def __init__(self, session_factory, profile_id: int):
        super().__init__(session_factory, profile_id)
        self._content_client = ContentClient(session_factory, profile_id)

    def record(  # cq: max-lines=40
        self,
        signal: SignalType,
        content_text: str | None = None,
        content_id: str | None = None,
        content_metadata: dict | None = None,
        strength: float = 1.0,
        tags: list[str] | None = None,
        comment: str | None = None,
        context: dict | None = None,
    ) -> int:
        """
        Record feedback on content.

        Args:
            signal: Feedback type (positive, negative, dismiss)
            content_text: Full text of content (stored for training)
            content_id: External content identifier
            content_metadata: Source-specific metadata
            strength: Signal strength 0.0-1.0 (default 1.0)
            tags: Optional categorization tags
            comment: Optional user comment
            context: Additional context (UI state, etc.)

        Returns:
            Created feedback record ID

        Raises:
            ValidationError: If inputs are invalid
        """
        # Validate signal
        if signal not in ("positive", "negative", "dismiss"):
            raise ValidationError(
                f"Invalid signal: {signal}. Must be positive, negative, or dismiss"
            )

        # Validate strength
        if strength < 0.0 or strength > 1.0:
            raise ValidationError(f"Strength must be between 0.0 and 1.0, got {strength}")

        with self._session_factory() as session:
            # Find or create content record if text provided
            db_content_id = None
            if content_text:
                db_content_id, _ = self._content_client.get_or_create(
                    content_text=content_text,
                    source="feedback",
                    external_id=content_id,
                    metadata=content_metadata,
                )

            # Create feedback record
            feedback = Feedback(
                profile_id=self.profile_id,
                content_id=db_content_id,
                signal=signal,
                strength=strength,
                tags=tags,
                comment=comment,
                context=context,
            )
            session.add(feedback)
            session.flush()

            return feedback.id

    def list_by_signal(
        self,
        signal: SignalType,
        limit: int = 100,
        since: datetime | None = None,
    ) -> list[Feedback]:
        """
        List feedback filtered by signal type for this profile.

        Args:
            signal: Signal type to filter by
            limit: Maximum records to return
            since: Only return feedback after this time

        Returns:
            List of feedback records
        """
        with self._session_factory() as session:
            stmt = select(Feedback).where(
                Feedback.profile_id == self.profile_id,
                Feedback.signal == signal,
            )

            if since:
                stmt = stmt.where(Feedback.created_at >= since)

            stmt = stmt.order_by(Feedback.created_at.desc()).limit(limit)
            objects = list(session.scalars(stmt).all())
            return cast(list[Feedback], detach_all(objects, session))

    def list_with_content(
        self,
        signal: SignalType | None = None,
        limit: int = 100,
        since: datetime | None = None,
    ) -> list[tuple[Feedback, Content | None]]:
        """
        List feedback with associated content for this profile.

        Args:
            signal: Optional signal type filter
            limit: Maximum records to return
            since: Only return feedback after this time

        Returns:
            List of (feedback, content) tuples
        """
        with self._session_factory() as session:
            stmt = (
                select(Feedback, Content)
                .outerjoin(Content, Feedback.content_id == Content.id)
                .where(Feedback.profile_id == self.profile_id)
            )

            if signal:
                stmt = stmt.where(Feedback.signal == signal)

            if since:
                stmt = stmt.where(Feedback.created_at >= since)

            stmt = stmt.order_by(Feedback.created_at.desc()).limit(limit)
            results = list(session.execute(stmt).all())
            # Detach both objects in each tuple
            for feedback, content in results:
                detach(feedback, session)
                detach(content, session)
            return results

    def count_by_signal(self) -> dict[str, int]:
        """
        Count feedback by signal type for this profile.

        Returns:
            Dict mapping signal type to count
        """
        with self._session_factory() as session:
            counts = {}
            for signal in ("positive", "negative", "dismiss"):
                stmt = select(Feedback).where(
                    Feedback.profile_id == self.profile_id,
                    Feedback.signal == signal,
                )
                count = len(list(session.scalars(stmt).all()))
                counts[signal] = count
            return counts
