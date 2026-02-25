"""Tests for feedback collection."""

import pytest

from llm_kelt import ValidationError


class TestFeedbackClient:
    """Test FeedbackClient functionality."""

    def test_record_feedback_positive(self, kelt_client, clean_tables):
        """Test recording positive feedback."""
        # Create content first
        content_id = kelt_client.content.create(
            content_text="Test article content",
            source="test",
        )

        fact_id = kelt_client.atomic.feedback.record(
            signal="positive",
            content_id=content_id,
            strength=0.9,
        )

        assert fact_id > 0

        fact = kelt_client.atomic.feedback.get(fact_id)
        assert fact is not None
        assert fact.feedback_details.signal == "positive"
        assert fact.feedback_details.strength == 0.9

    def test_record_feedback_negative(self, kelt_client, clean_tables):
        """Test recording negative feedback."""
        content_id = kelt_client.content.create(
            content_text="Test content",
            source="test",
        )

        fact_id = kelt_client.atomic.feedback.record(
            signal="negative",
            content_id=content_id,
            strength=0.5,
        )

        fact = kelt_client.atomic.feedback.get(fact_id)
        assert fact.feedback_details.signal == "negative"
        assert fact.feedback_details.strength == 0.5

    def test_record_feedback_dismiss(self, kelt_client, clean_tables):
        """Test recording dismiss feedback."""
        fact_id = kelt_client.atomic.feedback.record(
            signal="dismiss",
        )

        fact = kelt_client.atomic.feedback.get(fact_id)
        assert fact.feedback_details.signal == "dismiss"
        assert fact.feedback_details.strength == 1.0  # Default

    def test_record_feedback_with_tags(self, kelt_client, clean_tables):
        """Test feedback with tags."""
        fact_id = kelt_client.atomic.feedback.record(
            signal="negative",
            tags=["too_long", "off_topic"],
        )

        fact = kelt_client.atomic.feedback.get(fact_id)
        assert fact.feedback_details.tags == ["too_long", "off_topic"]

    def test_record_feedback_with_comment(self, kelt_client, clean_tables):
        """Test feedback with comment."""
        fact_id = kelt_client.atomic.feedback.record(
            signal="positive",
            comment="Very insightful article!",
        )

        fact = kelt_client.atomic.feedback.get(fact_id)
        assert fact.feedback_details.comment == "Very insightful article!"

    def test_invalid_signal_raises(self, kelt_client, clean_tables):
        """Test that invalid signal raises ValidationError."""
        with pytest.raises(ValidationError, match="Invalid signal"):
            kelt_client.atomic.feedback.record(
                signal="invalid",
            )

    def test_invalid_strength_too_low_raises(self, kelt_client, clean_tables):
        """Test that strength < 0 raises ValidationError."""
        with pytest.raises(ValidationError, match="strength must be"):
            kelt_client.atomic.feedback.record(
                signal="positive",
                strength=-0.1,
            )

    def test_invalid_strength_too_high_raises(self, kelt_client, clean_tables):
        """Test that strength > 1 raises ValidationError."""
        with pytest.raises(ValidationError, match="strength must be"):
            kelt_client.atomic.feedback.record(
                signal="positive",
                strength=1.5,
            )

    def test_list_by_signal(self, kelt_client, clean_tables):
        """Test listing feedback by signal type."""
        # Record multiple feedback items
        kelt_client.atomic.feedback.record(signal="positive")
        kelt_client.atomic.feedback.record(signal="negative")
        kelt_client.atomic.feedback.record(signal="positive")

        positive = kelt_client.atomic.feedback.list_by_signal("positive")
        assert len(positive) == 2

        negative = kelt_client.atomic.feedback.list_by_signal("negative")
        assert len(negative) == 1

    def test_count(self, kelt_client, clean_tables):
        """Test counting feedback."""
        assert kelt_client.atomic.feedback.count() == 0

        kelt_client.atomic.feedback.record(signal="positive")
        kelt_client.atomic.feedback.record(signal="negative")

        assert kelt_client.atomic.feedback.count() == 2

    def test_delete(self, kelt_client, clean_tables):
        """Test deleting feedback."""
        fact_id = kelt_client.atomic.feedback.record(
            signal="positive",
        )

        assert kelt_client.atomic.feedback.get(fact_id) is not None

        result = kelt_client.atomic.feedback.delete(fact_id)
        assert result is True

        assert kelt_client.atomic.feedback.get(fact_id) is None

    def test_feedback_with_content(self, kelt_client, clean_tables):
        """Test feedback linked to content."""
        content_id = kelt_client.content.create(
            content_text="Article content",
            source="test",
        )

        fact_id = kelt_client.atomic.feedback.record(
            signal="positive",
            content_id=content_id,
        )

        fact = kelt_client.atomic.feedback.get(fact_id)
        assert fact.feedback_details.content_id == content_id
