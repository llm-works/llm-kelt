"""Tests for feedback collection."""

import pytest

from llm_learn import ValidationError


class TestFeedbackClient:
    """Test FeedbackClient functionality."""

    def test_record_feedback_positive(self, learn_client, clean_tables):
        """Test recording positive feedback."""
        # Create content first
        content_id = learn_client.content.create(
            content_text="Test article content",
            source="test",
        )

        fact_id = learn_client.atomic.feedback.record(
            signal="positive",
            content_id=content_id,
            strength=0.9,
        )

        assert fact_id > 0

        fact = learn_client.atomic.feedback.get(fact_id)
        assert fact is not None
        assert fact.feedback_details.signal == "positive"
        assert fact.feedback_details.strength == 0.9

    def test_record_feedback_negative(self, learn_client, clean_tables):
        """Test recording negative feedback."""
        content_id = learn_client.content.create(
            content_text="Test content",
            source="test",
        )

        fact_id = learn_client.atomic.feedback.record(
            signal="negative",
            content_id=content_id,
            strength=0.5,
        )

        fact = learn_client.atomic.feedback.get(fact_id)
        assert fact.feedback_details.signal == "negative"
        assert fact.feedback_details.strength == 0.5

    def test_record_feedback_dismiss(self, learn_client, clean_tables):
        """Test recording dismiss feedback."""
        fact_id = learn_client.atomic.feedback.record(
            signal="dismiss",
        )

        fact = learn_client.atomic.feedback.get(fact_id)
        assert fact.feedback_details.signal == "dismiss"
        assert fact.feedback_details.strength == 1.0  # Default

    def test_record_feedback_with_tags(self, learn_client, clean_tables):
        """Test feedback with tags."""
        fact_id = learn_client.atomic.feedback.record(
            signal="negative",
            tags=["too_long", "off_topic"],
        )

        fact = learn_client.atomic.feedback.get(fact_id)
        assert fact.feedback_details.tags == ["too_long", "off_topic"]

    def test_record_feedback_with_comment(self, learn_client, clean_tables):
        """Test feedback with comment."""
        fact_id = learn_client.atomic.feedback.record(
            signal="positive",
            comment="Very insightful article!",
        )

        fact = learn_client.atomic.feedback.get(fact_id)
        assert fact.feedback_details.comment == "Very insightful article!"

    def test_invalid_signal_raises(self, learn_client, clean_tables):
        """Test that invalid signal raises ValidationError."""
        with pytest.raises(ValidationError, match="Invalid signal"):
            learn_client.atomic.feedback.record(
                signal="invalid",
            )

    def test_invalid_strength_too_low_raises(self, learn_client, clean_tables):
        """Test that strength < 0 raises ValidationError."""
        with pytest.raises(ValidationError, match="strength must be"):
            learn_client.atomic.feedback.record(
                signal="positive",
                strength=-0.1,
            )

    def test_invalid_strength_too_high_raises(self, learn_client, clean_tables):
        """Test that strength > 1 raises ValidationError."""
        with pytest.raises(ValidationError, match="strength must be"):
            learn_client.atomic.feedback.record(
                signal="positive",
                strength=1.5,
            )

    def test_list_by_signal(self, learn_client, clean_tables):
        """Test listing feedback by signal type."""
        # Record multiple feedback items
        learn_client.atomic.feedback.record(signal="positive")
        learn_client.atomic.feedback.record(signal="negative")
        learn_client.atomic.feedback.record(signal="positive")

        positive = learn_client.atomic.feedback.list_by_signal("positive")
        assert len(positive) == 2

        negative = learn_client.atomic.feedback.list_by_signal("negative")
        assert len(negative) == 1

    def test_count(self, learn_client, clean_tables):
        """Test counting feedback."""
        assert learn_client.atomic.feedback.count() == 0

        learn_client.atomic.feedback.record(signal="positive")
        learn_client.atomic.feedback.record(signal="negative")

        assert learn_client.atomic.feedback.count() == 2

    def test_delete(self, learn_client, clean_tables):
        """Test deleting feedback."""
        fact_id = learn_client.atomic.feedback.record(
            signal="positive",
        )

        assert learn_client.atomic.feedback.get(fact_id) is not None

        result = learn_client.atomic.feedback.delete(fact_id)
        assert result is True

        assert learn_client.atomic.feedback.get(fact_id) is None

    def test_feedback_with_content(self, learn_client, clean_tables):
        """Test feedback linked to content."""
        content_id = learn_client.content.create(
            content_text="Article content",
            source="test",
        )

        fact_id = learn_client.atomic.feedback.record(
            signal="positive",
            content_id=content_id,
        )

        fact = learn_client.atomic.feedback.get(fact_id)
        assert fact.feedback_details.content_id == content_id
