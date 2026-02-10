"""Predictions client for hypothesis tracking and calibration."""

from datetime import UTC, date, datetime
from typing import Literal, cast

from appinfra.db.utils import detach, detach_all
from sqlalchemy import select

from llm_learn.core.exceptions import NotFoundError, ValidationError

from ..models import Fact, PredictionDetails
from .base import FactClient

OutcomeType = Literal["correct", "incorrect", "partial", "cancelled"]


class PredictionsClient(FactClient[PredictionDetails]):
    """
    Client for tracking predictions and calibration.

    Predictions are hypotheses with confidence levels that can be
    resolved against actual outcomes for calibration tracking.

    Usage:
        predictions = PredictionsClient(session_factory, context_key="my-agent")

        # Record a prediction
        fact_id = predictions.record(
            hypothesis="BTC will exceed $100k by Q1 2025",
            confidence=0.65,
            resolution_date="2025-03-31",
            category="crypto",
            tags=["price", "bitcoin"],
        )

        # Resolve it
        predictions.resolve(
            fact_id=fact_id,
            outcome="correct",
            actual="BTC reached $105k on March 15",
        )
    """

    fact_type = "prediction"
    details_model = PredictionDetails
    details_relationship = "prediction_details"

    def _validate_prediction(self, hypothesis: str, confidence: float) -> None:
        """Validate prediction inputs."""
        if not hypothesis or not hypothesis.strip():
            raise ValidationError("hypothesis cannot be empty")
        if not (0.0 <= confidence <= 1.0):
            raise ValidationError(f"confidence must be between 0.0 and 1.0, got {confidence}")

    def _parse_resolution_date(self, resolution_date: date | str | None) -> date | None:
        """Parse resolution date from string or date object."""
        if resolution_date is None:
            return None
        if isinstance(resolution_date, str):
            try:
                return datetime.fromisoformat(resolution_date).date()
            except ValueError:
                raise ValidationError(f"Invalid date format: {resolution_date}")
        return resolution_date

    def _determine_resolution_type(
        self, parsed_date: date | None, event: str | None, metric: dict | None
    ) -> str | None:
        """Determine resolution type from provided criteria."""
        if parsed_date:
            return "date"
        return "event" if event else ("metric" if metric else None)

    def _create_prediction_fact(
        self, hypothesis: str, confidence: float, category: str | None
    ) -> Fact:
        """Create Fact record for prediction."""
        content_text = hypothesis.strip()
        return Fact(
            context_key=self.context_key,
            type=self.fact_type,
            content=content_text,
            content_hash=self._compute_content_hash(content_text),
            category=category.strip() if category else None,
            source="user",
            confidence=confidence,
            active=True,
        )

    def _create_prediction_details(
        self,
        fact_id: int,
        res_type: str | None,
        parsed_date: date | None,
        resolution_event: str | None,
        resolution_metric: dict | None,
        verification_source: str | None,
        verification_url: str | None,
        tags: list[str] | None,
    ) -> PredictionDetails:
        """Create PredictionDetails record."""
        return PredictionDetails(
            fact_id=fact_id,
            resolution_type=res_type,
            resolution_date=parsed_date,
            resolution_event=resolution_event,
            resolution_metric=resolution_metric,
            verification_source=verification_source,
            verification_url=verification_url,
            status="pending",
            tags=tags,
        )

    def record(
        self,
        hypothesis: str,
        confidence: float,
        resolution_date: date | str | None = None,
        resolution_event: str | None = None,
        resolution_metric: dict | None = None,
        category: str | None = None,
        tags: list[str] | None = None,
        verification_source: str | None = None,
        verification_url: str | None = None,
    ) -> int:
        """Record a prediction."""
        self._validate_prediction(hypothesis, confidence)
        parsed_date = self._parse_resolution_date(resolution_date)
        res_type = self._determine_resolution_type(parsed_date, resolution_event, resolution_metric)

        with self._session_factory() as session:
            fact = self._create_prediction_fact(hypothesis, confidence, category)
            session.add(fact)
            session.flush()

            details = self._create_prediction_details(
                fact.id,
                res_type,
                parsed_date,
                resolution_event,
                resolution_metric,
                verification_source,
                verification_url,
                tags,
            )
            session.add(details)
            session.flush()

            # Auto-embed if embedder configured
            self._auto_embed_fact(fact, session)

            return fact.id

    def resolve(
        self,
        fact_id: int,
        outcome: OutcomeType,
        actual: str | None = None,
        outcome_confidence: float | None = None,
    ) -> bool:
        """Resolve a prediction with an outcome."""
        valid_outcomes = ("correct", "incorrect", "partial", "cancelled")
        if outcome not in valid_outcomes:
            raise ValidationError(f"Invalid outcome: {outcome}. Must be one of {valid_outcomes}")

        if outcome_confidence is not None and not (0.0 <= outcome_confidence <= 1.0):
            raise ValidationError(
                f"outcome_confidence must be between 0.0 and 1.0, got {outcome_confidence}"
            )

        with self._session_factory() as session:
            fact = self._get_fact(session, fact_id)
            if fact is None:
                raise NotFoundError(f"Prediction {fact_id} not found")

            details = fact.prediction_details
            if details is None:
                raise NotFoundError(f"Prediction details for {fact_id} not found")

            if details.status == "resolved":
                raise ValidationError(f"Prediction {fact_id} is already resolved")

            details.status = "resolved"
            details.outcome = outcome
            details.outcome_confidence = outcome_confidence
            details.actual_result = actual
            details.resolved_at = datetime.now(UTC)

            return True

    def list_pending(
        self,
        category: str | None = None,
        limit: int = 100,
    ) -> list[Fact]:
        """List pending (unresolved) predictions."""
        with self._session_factory() as session:
            stmt = (
                select(Fact)
                .join(PredictionDetails)
                .where(
                    Fact.context_key == self.context_key,
                    Fact.type == self.fact_type,
                    Fact.active == True,  # noqa: E712
                    PredictionDetails.status == "pending",
                )
            )

            if category:
                stmt = stmt.where(Fact.category == category)

            stmt = stmt.order_by(PredictionDetails.resolution_date.asc().nullslast()).limit(limit)

            facts = list(session.scalars(stmt).all())
            for fact in facts:
                details = fact.prediction_details
                if details is not None:
                    detach(details, session)
            return cast(list[Fact], detach_all(facts, session))

    def list_resolved(
        self,
        outcome: OutcomeType | None = None,
        category: str | None = None,
        since: datetime | None = None,
        limit: int = 100,
    ) -> list[Fact]:
        """List resolved predictions."""
        with self._session_factory() as session:
            stmt = (
                select(Fact)
                .join(PredictionDetails)
                .where(
                    Fact.context_key == self.context_key,
                    Fact.type == self.fact_type,
                    Fact.active == True,  # noqa: E712
                    PredictionDetails.status == "resolved",
                )
            )

            if outcome:
                stmt = stmt.where(PredictionDetails.outcome == outcome)

            if category:
                stmt = stmt.where(Fact.category == category)

            if since:
                stmt = stmt.where(PredictionDetails.resolved_at >= since)

            stmt = stmt.order_by(PredictionDetails.resolved_at.desc()).limit(limit)

            facts = list(session.scalars(stmt).all())
            for fact in facts:
                details = fact.prediction_details
                if details is not None:
                    detach(details, session)
            return cast(list[Fact], detach_all(facts, session))

    def _compute_bucket_stats(
        self, results: list[tuple[Fact, PredictionDetails]]
    ) -> dict[str, dict[str, int | float]]:
        """Aggregate results into confidence buckets with accuracy stats."""
        buckets: dict[str, dict[str, int]] = {}
        for fact, details in results:
            bucket = f"{int(fact.confidence * 10) / 10:.1f}"
            if bucket not in buckets:
                buckets[bucket] = {"total": 0, "correct": 0}
            buckets[bucket]["total"] += 1
            if details.outcome == "correct":
                buckets[bucket]["correct"] += 1

        return {
            bucket: {
                "total": counts["total"],
                "correct": counts["correct"],
                "accuracy": counts["correct"] / counts["total"] if counts["total"] > 0 else 0.0,
            }
            for bucket, counts in sorted(buckets.items())
        }

    def get_calibration_stats(
        self, category: str | None = None
    ) -> dict[str, dict[str, int | float]]:
        """Get calibration statistics by confidence bucket."""
        with self._session_factory() as session:
            stmt = (
                select(Fact, PredictionDetails)
                .join(PredictionDetails)
                .where(
                    Fact.context_key == self.context_key,
                    Fact.type == self.fact_type,
                    Fact.active == True,  # noqa: E712
                    PredictionDetails.status == "resolved",
                    PredictionDetails.outcome.in_(["correct", "incorrect"]),
                )
            )
            if category:
                stmt = stmt.where(Fact.category == category)
            results = session.execute(stmt).all()
            return self._compute_bucket_stats(results)
