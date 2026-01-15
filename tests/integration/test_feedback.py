"""Tests for feedback collection."""

import pytest

from llm_learn import ValidationError


class TestFeedbackClient:
    """Test FeedbackClient functionality."""

    def test_record_feedback_positive(self, learn_client, clean_tables):
        """Test recording positive feedback."""
        feedback_id = learn_client.feedback.record(
            content_text="Test article content",
            signal="positive",
            strength=0.9,
        )

        assert feedback_id > 0

        feedback = learn_client.feedback.get(feedback_id)
        assert feedback is not None
        assert feedback.signal == "positive"
        assert feedback.strength == 0.9

    def test_record_feedback_negative(self, learn_client, clean_tables):
        """Test recording negative feedback."""
        feedback_id = learn_client.feedback.record(
            content_text="Test content",
            signal="negative",
            strength=0.5,
        )

        feedback = learn_client.feedback.get(feedback_id)
        assert feedback.signal == "negative"
        assert feedback.strength == 0.5

    def test_record_feedback_dismiss(self, learn_client, clean_tables):
        """Test recording dismiss feedback."""
        feedback_id = learn_client.feedback.record(
            content_text="Test content",
            signal="dismiss",
        )

        feedback = learn_client.feedback.get(feedback_id)
        assert feedback.signal == "dismiss"
        assert feedback.strength == 1.0  # Default

    def test_record_feedback_with_tags(self, learn_client, clean_tables):
        """Test feedback with tags."""
        feedback_id = learn_client.feedback.record(
            content_text="Test content",
            signal="negative",
            tags=["too_long", "off_topic"],
        )

        feedback = learn_client.feedback.get(feedback_id)
        assert feedback.tags == ["too_long", "off_topic"]

    def test_record_feedback_with_comment(self, learn_client, clean_tables):
        """Test feedback with comment."""
        feedback_id = learn_client.feedback.record(
            content_text="Test content",
            signal="positive",
            comment="Very insightful article!",
        )

        feedback = learn_client.feedback.get(feedback_id)
        assert feedback.comment == "Very insightful article!"

    def test_invalid_signal_raises(self, learn_client, clean_tables):
        """Test that invalid signal raises ValidationError."""
        with pytest.raises(ValidationError, match="Invalid signal"):
            learn_client.feedback.record(
                content_text="Test",
                signal="invalid",
            )

    def test_invalid_strength_too_low_raises(self, learn_client, clean_tables):
        """Test that strength < 0 raises ValidationError."""
        with pytest.raises(ValidationError, match="Strength must be"):
            learn_client.feedback.record(
                content_text="Test",
                signal="positive",
                strength=-0.1,
            )

    def test_invalid_strength_too_high_raises(self, learn_client, clean_tables):
        """Test that strength > 1 raises ValidationError."""
        with pytest.raises(ValidationError, match="Strength must be"):
            learn_client.feedback.record(
                content_text="Test",
                signal="positive",
                strength=1.5,
            )

    def test_list_by_signal(self, learn_client, clean_tables):
        """Test listing feedback by signal type."""
        # Record multiple feedback items
        learn_client.feedback.record(content_text="A", signal="positive")
        learn_client.feedback.record(content_text="B", signal="negative")
        learn_client.feedback.record(content_text="C", signal="positive")

        positive = learn_client.feedback.list_by_signal("positive")
        assert len(positive) == 2

        negative = learn_client.feedback.list_by_signal("negative")
        assert len(negative) == 1

    def test_count(self, learn_client, clean_tables):
        """Test counting feedback."""
        assert learn_client.feedback.count() == 0

        learn_client.feedback.record(content_text="A", signal="positive")
        learn_client.feedback.record(content_text="B", signal="negative")

        assert learn_client.feedback.count() == 2

    def test_delete(self, learn_client, clean_tables):
        """Test deleting feedback."""
        feedback_id = learn_client.feedback.record(
            content_text="Test",
            signal="positive",
        )

        assert learn_client.feedback.exists(feedback_id)

        result = learn_client.feedback.delete(feedback_id)
        assert result is True

        assert not learn_client.feedback.exists(feedback_id)

    def test_content_deduplication(self, learn_client, clean_tables):
        """Test that same content is not duplicated."""
        content_text = "Same content for both feedbacks"

        learn_client.feedback.record(content_text=content_text, signal="positive")
        learn_client.feedback.record(content_text=content_text, signal="negative")

        # Should have 2 feedback records but only 1 content record
        assert learn_client.feedback.count() == 2
        assert learn_client.content.count() == 1
