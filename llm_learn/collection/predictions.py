"""Predictions collection client."""

from datetime import UTC, date, datetime
from typing import Literal, cast

from appinfra.db.utils import detach_all
from sqlalchemy import select

from ..core.exceptions import NotFoundError, ValidationError
from ..core.models import Prediction
from .base import ProfileScopedClient

OutcomeType = Literal["correct", "incorrect", "partial", "cancelled"]


class PredictionsClient(ProfileScopedClient[Prediction]):
    """
    Client for tracking predictions and calibration scoped to a profile.

    Predictions are hypotheses with confidence levels that can be
    resolved against actual outcomes for calibration tracking.

    Usage:
        predictions = PredictionsClient(session_factory, profile_id=123)
        pred_id = predictions.record(
            hypothesis="BTC will exceed $100k by Q1 2025",
            confidence=0.65,
            resolution_date="2025-03-31",
            domain="crypto",
            tags=["price", "bitcoin"],
        )

        predictions.resolve(
            prediction_id=pred_id,
            outcome="correct",
            actual="BTC reached $105k on March 15",
        )
    """

    model = Prediction

    def record(  # cq: max-lines=50
        self,
        hypothesis: str,
        confidence: float,
        resolution_date: date | str | None = None,
        resolution_event: str | None = None,
        resolution_metric: dict | None = None,
        domain: str | None = None,
        tags: list[str] | None = None,
        confidence_reasoning: str | None = None,
        verification_source: str | None = None,
        verification_url: str | None = None,
    ) -> int:
        """
        Record a prediction.

        Args:
            hypothesis: The prediction statement
            confidence: Confidence level 0.0-1.0
            resolution_date: When prediction can be verified
            resolution_event: Event that triggers resolution
            resolution_metric: Metric-based resolution criteria
            domain: Topic area
            tags: Categorization tags
            confidence_reasoning: Explanation of confidence
            verification_source: How to verify (polymarket, news, manual)
            verification_url: URL for verification

        Returns:
            Created prediction ID

        Raises:
            ValidationError: If inputs are invalid
        """
        # Validate inputs
        if not hypothesis or not hypothesis.strip():
            raise ValidationError("Hypothesis cannot be empty")

        if confidence < 0.0 or confidence > 1.0:
            raise ValidationError(f"Confidence must be between 0.0 and 1.0, got {confidence}")

        # Parse date string if needed
        parsed_date: date | None = None
        if resolution_date is not None:
            if isinstance(resolution_date, str):
                try:
                    parsed_date = datetime.fromisoformat(resolution_date).date()
                except ValueError:
                    raise ValidationError(f"Invalid date format: {resolution_date}")
            else:
                parsed_date = resolution_date

        # Determine resolution type
        resolution_type = None
        if parsed_date:
            resolution_type = "date"
        elif resolution_event:
            resolution_type = "event"
        elif resolution_metric:
            resolution_type = "metric"

        with self._session_factory() as session:
            prediction = Prediction(
                profile_id=self.profile_id,
                hypothesis=hypothesis.strip(),
                confidence=confidence,
                confidence_reasoning=confidence_reasoning,
                resolution_type=resolution_type,
                resolution_date=parsed_date,
                resolution_event=resolution_event,
                resolution_metric=resolution_metric,
                verification_source=verification_source,
                verification_url=verification_url,
                domain=domain.strip() if domain else None,
                tags=tags,
            )
            session.add(prediction)
            session.flush()

            return prediction.id

    def resolve(
        self,
        prediction_id: int,
        outcome: OutcomeType,
        actual: str | None = None,
        outcome_confidence: float | None = None,
    ) -> bool:
        """
        Resolve a prediction with an outcome.

        Args:
            prediction_id: ID of prediction to resolve
            outcome: Resolution outcome (correct, incorrect, partial, cancelled)
            actual: What actually happened
            outcome_confidence: Confidence in outcome assessment (0.0-1.0)

        Returns:
            True if resolved successfully

        Raises:
            ValidationError: If inputs are invalid
            NotFoundError: If prediction not found or belongs to different profile
        """
        # Validate outcome
        valid_outcomes = ("correct", "incorrect", "partial", "cancelled")
        if outcome not in valid_outcomes:
            raise ValidationError(f"Invalid outcome: {outcome}. Must be one of {valid_outcomes}")

        # Validate outcome confidence
        if outcome_confidence is not None and (
            outcome_confidence < 0.0 or outcome_confidence > 1.0
        ):
            raise ValidationError(
                f"Outcome confidence must be between 0.0 and 1.0, got {outcome_confidence}"
            )

        with self._session_factory() as session:
            prediction = session.get(Prediction, prediction_id)
            if not prediction or prediction.profile_id != self.profile_id:
                raise NotFoundError(f"Prediction {prediction_id} not found")

            prediction.outcome = outcome
            prediction.actual_result = actual
            prediction.outcome_confidence = outcome_confidence
            prediction.status = "resolved"
            prediction.resolved_at = datetime.now(UTC)

            return True

    def list_pending(self, limit: int = 100) -> list[Prediction]:
        """
        List pending predictions for this profile.

        Args:
            limit: Maximum records to return

        Returns:
            List of pending predictions, ordered by resolution date
        """
        with self._session_factory() as session:
            stmt = (
                select(Prediction)
                .where(
                    Prediction.profile_id == self.profile_id,
                    Prediction.status == "pending",
                )
                .order_by(Prediction.resolution_date.asc().nullslast())
                .limit(limit)
            )
            objects = list(session.scalars(stmt).all())
            return cast(list[Prediction], detach_all(objects, session))

    def list_due(self, as_of: date | None = None) -> list[Prediction]:
        """
        List predictions due for resolution for this profile.

        Args:
            as_of: Date to check against (default: today)

        Returns:
            List of predictions past their resolution date
        """
        if as_of is None:
            as_of = date.today()

        with self._session_factory() as session:
            stmt = (
                select(Prediction)
                .where(
                    Prediction.profile_id == self.profile_id,
                    Prediction.status == "pending",
                    Prediction.resolution_date <= as_of,
                )
                .order_by(Prediction.resolution_date.asc())
            )
            objects = list(session.scalars(stmt).all())
            return cast(list[Prediction], detach_all(objects, session))

    def list_resolved(
        self,
        outcome: OutcomeType | None = None,
        limit: int = 100,
        since: datetime | None = None,
    ) -> list[Prediction]:
        """
        List resolved predictions for this profile.

        Args:
            outcome: Optional outcome filter
            limit: Maximum records to return
            since: Only return predictions resolved after this time

        Returns:
            List of resolved predictions
        """
        with self._session_factory() as session:
            stmt = select(Prediction).where(
                Prediction.profile_id == self.profile_id,
                Prediction.status == "resolved",
            )

            if outcome:
                stmt = stmt.where(Prediction.outcome == outcome)

            if since:
                stmt = stmt.where(Prediction.resolved_at >= since)

            stmt = stmt.order_by(Prediction.resolved_at.desc()).limit(limit)
            objects = list(session.scalars(stmt).all())
            return cast(list[Prediction], detach_all(objects, session))

    def list_by_domain(
        self,
        domain: str,
        status: str | None = None,
        limit: int = 100,
    ) -> list[Prediction]:
        """
        List predictions by domain for this profile.

        Args:
            domain: Domain to filter by
            status: Optional status filter (pending, resolved)
            limit: Maximum records to return

        Returns:
            List of predictions
        """
        with self._session_factory() as session:
            stmt = select(Prediction).where(
                Prediction.profile_id == self.profile_id,
                Prediction.domain == domain,
            )

            if status:
                stmt = stmt.where(Prediction.status == status)

            stmt = stmt.order_by(Prediction.created_at.desc()).limit(limit)
            objects = list(session.scalars(stmt).all())
            return cast(list[Prediction], detach_all(objects, session))

    def get_calibration_data(
        self,
        domain: str | None = None,
        since: datetime | None = None,
    ) -> list[tuple[float, bool]]:
        """
        Get calibration data (confidence, was_correct) pairs for this profile.

        Args:
            domain: Optional domain filter
            since: Only include predictions resolved after this time

        Returns:
            List of (confidence, was_correct) tuples
        """
        with self._session_factory() as session:
            stmt = select(Prediction).where(
                Prediction.profile_id == self.profile_id,
                Prediction.status == "resolved",
                Prediction.outcome.in_(["correct", "incorrect"]),
            )

            if domain:
                stmt = stmt.where(Prediction.domain == domain)

            if since:
                stmt = stmt.where(Prediction.resolved_at >= since)

            predictions = list(session.scalars(stmt).all())
            return [(p.confidence, p.outcome == "correct") for p in predictions]
