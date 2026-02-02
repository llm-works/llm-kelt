"""Tests for predictions collection."""

from datetime import date, timedelta

import pytest

from llm_learn import NotFoundError, ValidationError


class TestPredictionsClient:
    """Test PredictionsClient functionality."""

    def test_record_prediction(self, learn_client, clean_tables):
        """Test recording a prediction."""
        fact_id = learn_client.predictions.record(
            hypothesis="BTC will exceed $100k by Q1 2025",
            confidence=0.65,
            resolution_date="2025-03-31",
            category="crypto",
            tags=["price", "bitcoin"],
        )

        assert fact_id > 0

        fact = learn_client.predictions.get(fact_id)
        assert fact is not None
        assert fact.content == "BTC will exceed $100k by Q1 2025"
        assert fact.confidence == 0.65
        assert fact.prediction_details.resolution_date == date(2025, 3, 31)
        assert fact.category == "crypto"
        assert fact.prediction_details.tags == ["price", "bitcoin"]
        assert fact.prediction_details.status == "pending"

    def test_record_with_date_object(self, learn_client, clean_tables):
        """Test recording with date object."""
        fact_id = learn_client.predictions.record(
            hypothesis="Test prediction",
            confidence=0.5,
            resolution_date=date(2025, 6, 15),
        )

        fact = learn_client.predictions.get(fact_id)
        assert fact.prediction_details.resolution_date == date(2025, 6, 15)
        assert fact.prediction_details.resolution_type == "date"

    def test_record_with_event(self, learn_client, clean_tables):
        """Test recording with event-based resolution."""
        fact_id = learn_client.predictions.record(
            hypothesis="Company X will IPO",
            confidence=0.4,
            resolution_event="IPO announcement",
        )

        fact = learn_client.predictions.get(fact_id)
        assert fact.prediction_details.resolution_type == "event"
        assert fact.prediction_details.resolution_event == "IPO announcement"

    def test_record_with_metric(self, learn_client, clean_tables):
        """Test recording with metric-based resolution."""
        fact_id = learn_client.predictions.record(
            hypothesis="Stock will hit target",
            confidence=0.6,
            resolution_metric={"metric": "AAPL_price", "operator": ">", "value": 200},
        )

        fact = learn_client.predictions.get(fact_id)
        assert fact.prediction_details.resolution_type == "metric"
        assert fact.prediction_details.resolution_metric["metric"] == "AAPL_price"

    def test_empty_hypothesis_raises(self, learn_client, clean_tables):
        """Test that empty hypothesis raises ValidationError."""
        with pytest.raises(ValidationError, match="hypothesis cannot be empty"):
            learn_client.predictions.record(
                hypothesis="",
                confidence=0.5,
            )

    def test_invalid_confidence_raises(self, learn_client, clean_tables):
        """Test that invalid confidence raises ValidationError."""
        with pytest.raises(ValidationError, match="confidence must be"):
            learn_client.predictions.record(
                hypothesis="Test",
                confidence=1.5,
            )

    def test_resolve_correct(self, learn_client, clean_tables):
        """Test resolving prediction as correct."""
        fact_id = learn_client.predictions.record(
            hypothesis="Test prediction",
            confidence=0.7,
        )

        result = learn_client.predictions.resolve(
            fact_id=fact_id,
            outcome="correct",
            actual="It happened as predicted",
        )

        assert result is True

        fact = learn_client.predictions.get(fact_id)
        assert fact.prediction_details.status == "resolved"
        assert fact.prediction_details.outcome == "correct"
        assert fact.prediction_details.actual_result == "It happened as predicted"
        assert fact.prediction_details.resolved_at is not None

    def test_resolve_incorrect(self, learn_client, clean_tables):
        """Test resolving prediction as incorrect."""
        fact_id = learn_client.predictions.record(
            hypothesis="Test prediction",
            confidence=0.7,
        )

        learn_client.predictions.resolve(
            fact_id=fact_id,
            outcome="incorrect",
            actual="It did not happen",
        )

        fact = learn_client.predictions.get(fact_id)
        assert fact.prediction_details.outcome == "incorrect"

    def test_resolve_partial(self, learn_client, clean_tables):
        """Test resolving prediction as partial."""
        fact_id = learn_client.predictions.record(
            hypothesis="Test prediction",
            confidence=0.7,
        )

        learn_client.predictions.resolve(
            fact_id=fact_id,
            outcome="partial",
            outcome_confidence=0.6,
        )

        fact = learn_client.predictions.get(fact_id)
        assert fact.prediction_details.outcome == "partial"
        assert fact.prediction_details.outcome_confidence == 0.6

    def test_resolve_not_found_raises(self, learn_client, clean_tables):
        """Test that resolving non-existent prediction raises NotFoundError."""
        with pytest.raises(NotFoundError):
            learn_client.predictions.resolve(
                fact_id=99999,
                outcome="correct",
            )

    def test_list_pending(self, learn_client, clean_tables):
        """Test listing pending predictions."""
        learn_client.predictions.record(hypothesis="A", confidence=0.5)
        fact_id = learn_client.predictions.record(hypothesis="B", confidence=0.6)
        learn_client.predictions.record(hypothesis="C", confidence=0.7)

        # Resolve one
        learn_client.predictions.resolve(fact_id, "correct")

        pending = learn_client.predictions.list_pending()
        assert len(pending) == 2

    def test_list_due(self, learn_client, clean_tables):
        """Test listing due predictions."""
        yesterday = date.today() - timedelta(days=1)
        tomorrow = date.today() + timedelta(days=1)

        learn_client.predictions.record(
            hypothesis="Past due",
            confidence=0.5,
            resolution_date=yesterday,
        )
        learn_client.predictions.record(
            hypothesis="Not due yet",
            confidence=0.5,
            resolution_date=tomorrow,
        )

        # list_pending returns all pending sorted by resolution_date
        # Due predictions are those with resolution_date <= today
        pending = learn_client.predictions.list_pending()
        # Find the one that's past due
        due = [
            p
            for p in pending
            if p.prediction_details.resolution_date
            and p.prediction_details.resolution_date <= date.today()
        ]
        assert len(due) == 1
        assert due[0].content == "Past due"

    def test_get_calibration_stats(self, learn_client, clean_tables):
        """Test getting calibration statistics."""
        # Record and resolve some predictions
        for conf, outcome in [(0.3, "incorrect"), (0.7, "correct"), (0.9, "correct")]:
            fact_id = learn_client.predictions.record(
                hypothesis=f"Test {conf}",
                confidence=conf,
            )
            learn_client.predictions.resolve(fact_id, outcome)

        stats = learn_client.predictions.get_calibration_stats()
        # Stats are bucketed by confidence (0.3, 0.7, 0.9)
        assert "0.3" in stats
        assert "0.7" in stats
        assert "0.9" in stats
        assert stats["0.3"]["total"] == 1
        assert stats["0.3"]["correct"] == 0
        assert stats["0.7"]["correct"] == 1
        assert stats["0.9"]["correct"] == 1
