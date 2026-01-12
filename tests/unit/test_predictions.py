"""Tests for predictions collection."""

from datetime import date, timedelta

import pytest

from llm_learn import NotFoundError, ValidationError


class TestPredictionsClient:
    """Test PredictionsClient functionality."""

    def test_record_prediction(self, learn_client, clean_tables):
        """Test recording a prediction."""
        pred_id = learn_client.predictions.record(
            hypothesis="BTC will exceed $100k by Q1 2025",
            confidence=0.65,
            resolution_date="2025-03-31",
            domain="crypto",
            tags=["price", "bitcoin"],
        )

        assert pred_id > 0

        pred = learn_client.predictions.get(pred_id)
        assert pred is not None
        assert pred.hypothesis == "BTC will exceed $100k by Q1 2025"
        assert pred.confidence == 0.65
        assert pred.resolution_date == date(2025, 3, 31)
        assert pred.domain == "crypto"
        assert pred.tags == ["price", "bitcoin"]
        assert pred.status == "pending"

    def test_record_with_date_object(self, learn_client, clean_tables):
        """Test recording with date object."""
        pred_id = learn_client.predictions.record(
            hypothesis="Test prediction",
            confidence=0.5,
            resolution_date=date(2025, 6, 15),
        )

        pred = learn_client.predictions.get(pred_id)
        assert pred.resolution_date == date(2025, 6, 15)
        assert pred.resolution_type == "date"

    def test_record_with_event(self, learn_client, clean_tables):
        """Test recording with event-based resolution."""
        pred_id = learn_client.predictions.record(
            hypothesis="Company X will IPO",
            confidence=0.4,
            resolution_event="IPO announcement",
        )

        pred = learn_client.predictions.get(pred_id)
        assert pred.resolution_type == "event"
        assert pred.resolution_event == "IPO announcement"

    def test_record_with_metric(self, learn_client, clean_tables):
        """Test recording with metric-based resolution."""
        pred_id = learn_client.predictions.record(
            hypothesis="Stock will hit target",
            confidence=0.6,
            resolution_metric={"metric": "AAPL_price", "operator": ">", "value": 200},
        )

        pred = learn_client.predictions.get(pred_id)
        assert pred.resolution_type == "metric"
        assert pred.resolution_metric["metric"] == "AAPL_price"

    def test_empty_hypothesis_raises(self, learn_client, clean_tables):
        """Test that empty hypothesis raises ValidationError."""
        with pytest.raises(ValidationError, match="Hypothesis cannot be empty"):
            learn_client.predictions.record(
                hypothesis="",
                confidence=0.5,
            )

    def test_invalid_confidence_raises(self, learn_client, clean_tables):
        """Test that invalid confidence raises ValidationError."""
        with pytest.raises(ValidationError, match="Confidence must be"):
            learn_client.predictions.record(
                hypothesis="Test",
                confidence=1.5,
            )

    def test_resolve_correct(self, learn_client, clean_tables):
        """Test resolving prediction as correct."""
        pred_id = learn_client.predictions.record(
            hypothesis="Test prediction",
            confidence=0.7,
        )

        result = learn_client.predictions.resolve(
            prediction_id=pred_id,
            outcome="correct",
            actual="It happened as predicted",
        )

        assert result is True

        pred = learn_client.predictions.get(pred_id)
        assert pred.status == "resolved"
        assert pred.outcome == "correct"
        assert pred.actual_result == "It happened as predicted"
        assert pred.resolved_at is not None

    def test_resolve_incorrect(self, learn_client, clean_tables):
        """Test resolving prediction as incorrect."""
        pred_id = learn_client.predictions.record(
            hypothesis="Test prediction",
            confidence=0.7,
        )

        learn_client.predictions.resolve(
            prediction_id=pred_id,
            outcome="incorrect",
            actual="It did not happen",
        )

        pred = learn_client.predictions.get(pred_id)
        assert pred.outcome == "incorrect"

    def test_resolve_partial(self, learn_client, clean_tables):
        """Test resolving prediction as partial."""
        pred_id = learn_client.predictions.record(
            hypothesis="Test prediction",
            confidence=0.7,
        )

        learn_client.predictions.resolve(
            prediction_id=pred_id,
            outcome="partial",
            outcome_confidence=0.6,
        )

        pred = learn_client.predictions.get(pred_id)
        assert pred.outcome == "partial"
        assert pred.outcome_confidence == 0.6

    def test_resolve_not_found_raises(self, learn_client, clean_tables):
        """Test that resolving non-existent prediction raises NotFoundError."""
        with pytest.raises(NotFoundError):
            learn_client.predictions.resolve(
                prediction_id=99999,
                outcome="correct",
            )

    def test_list_pending(self, learn_client, clean_tables):
        """Test listing pending predictions."""
        learn_client.predictions.record(hypothesis="A", confidence=0.5)
        pred_id = learn_client.predictions.record(hypothesis="B", confidence=0.6)
        learn_client.predictions.record(hypothesis="C", confidence=0.7)

        # Resolve one
        learn_client.predictions.resolve(pred_id, "correct")

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

        due = learn_client.predictions.list_due()
        assert len(due) == 1
        assert due[0].hypothesis == "Past due"

    def test_get_calibration_data(self, learn_client, clean_tables):
        """Test getting calibration data."""
        # Record and resolve some predictions
        for conf, outcome in [(0.3, "incorrect"), (0.7, "correct"), (0.9, "correct")]:
            pred_id = learn_client.predictions.record(
                hypothesis=f"Test {conf}",
                confidence=conf,
            )
            learn_client.predictions.resolve(pred_id, outcome)

        calibration = learn_client.predictions.get_calibration_data()
        assert len(calibration) == 3
        assert (0.3, False) in calibration
        assert (0.7, True) in calibration
        assert (0.9, True) in calibration
