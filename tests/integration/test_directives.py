"""Tests for directives collection."""

from datetime import UTC, datetime, timedelta

import pytest

from llm_kelt import ValidationError


class TestDirectivesClient:
    """Test DirectivesClient functionality."""

    def test_record_directive(self, learn_client, clean_tables):
        """Test recording a directive."""
        fact_id = learn_client.atomic.directives.record(
            text="Focus more on AI safety research",
            directive_type="standing",
        )

        assert fact_id > 0

        fact = learn_client.atomic.directives.get(fact_id)
        assert fact is not None
        assert fact.content == "Focus more on AI safety research"
        assert fact.directive_details.directive_type == "standing"
        assert fact.directive_details.status == "active"

    def test_record_one_time(self, learn_client, clean_tables):
        """Test recording a one-time directive."""
        fact_id = learn_client.atomic.directives.record(
            text="Research vector databases",
            directive_type="one-time",
        )

        fact = learn_client.atomic.directives.get(fact_id)
        assert fact.directive_details.directive_type == "one-time"

    def test_record_rule(self, learn_client, clean_tables):
        """Test recording a rule directive."""
        fact_id = learn_client.atomic.directives.record(
            text="Never show crypto price news",
            directive_type="rule",
            parsed_rules={"action": "exclude", "topic": "crypto_prices"},
        )

        fact = learn_client.atomic.directives.get(fact_id)
        assert fact.directive_details.directive_type == "rule"
        assert fact.directive_details.parsed_rules["action"] == "exclude"

    def test_record_with_expiration(self, learn_client, clean_tables):
        """Test recording directive with expiration."""
        expires = datetime.now(UTC) + timedelta(days=30)

        fact_id = learn_client.atomic.directives.record(
            text="Temporary directive",
            expires_at=expires,
        )

        fact = learn_client.atomic.directives.get(fact_id)
        assert fact.directive_details.expires_at is not None

    def test_empty_text_raises(self, learn_client, clean_tables):
        """Test that empty text raises ValidationError."""
        with pytest.raises(ValidationError, match="text cannot be empty"):
            learn_client.atomic.directives.record(text="")

    def test_invalid_type_raises(self, learn_client, clean_tables):
        """Test that invalid type raises ValidationError."""
        with pytest.raises(ValidationError, match="Invalid directive_type"):
            learn_client.atomic.directives.record(
                text="Test",
                directive_type="invalid",
            )

    def test_set_status(self, learn_client, clean_tables):
        """Test updating directive status."""
        fact_id = learn_client.atomic.directives.record(text="Test directive")

        # Pause
        result = learn_client.atomic.directives.set_status(fact_id, "paused")
        assert result is True

        fact = learn_client.atomic.directives.get(fact_id)
        assert fact.directive_details.status == "paused"

        # Complete
        learn_client.atomic.directives.set_status(fact_id, "completed")
        fact = learn_client.atomic.directives.get(fact_id)
        assert fact.directive_details.status == "completed"

    def test_set_status_not_found(self, learn_client, clean_tables):
        """Test that setting status on non-existent directive returns False."""
        # set_status returns False if not found, doesn't raise
        result = learn_client.atomic.directives.set_status(99999, "paused")
        assert result is False

    def test_set_status_invalid_raises(self, learn_client, clean_tables):
        """Test that invalid status raises ValidationError."""
        fact_id = learn_client.atomic.directives.record(text="Test")

        with pytest.raises(ValidationError, match="Invalid status"):
            learn_client.atomic.directives.set_status(fact_id, "invalid")

    def test_list_active(self, learn_client, clean_tables):
        """Test listing active directives."""
        learn_client.atomic.directives.record(text="Active 1")
        fact_id = learn_client.atomic.directives.record(text="To be paused")
        learn_client.atomic.directives.record(text="Active 2")

        # Pause one
        learn_client.atomic.directives.set_status(fact_id, "paused")

        active = learn_client.atomic.directives.list_active()
        assert len(active) == 2

    def test_list_active_filters_expired(self, learn_client, clean_tables):
        """Test that list_active filters out expired directives."""
        past = datetime.now(UTC) - timedelta(days=1)
        future = datetime.now(UTC) + timedelta(days=1)

        learn_client.atomic.directives.record(text="Expired", expires_at=past)
        learn_client.atomic.directives.record(text="Not expired", expires_at=future)
        learn_client.atomic.directives.record(text="No expiration")

        active = learn_client.atomic.directives.list_active()
        assert len(active) == 2
        texts = [d.content for d in active]
        assert "Expired" not in texts

    def test_list_by_type(self, learn_client, clean_tables):
        """Test listing directives by type."""
        learn_client.atomic.directives.record(text="A", directive_type="standing")
        learn_client.atomic.directives.record(text="B", directive_type="rule")
        learn_client.atomic.directives.record(text="C", directive_type="standing")

        standing = learn_client.atomic.directives.list_by_type("standing")
        assert len(standing) == 2

        rules = learn_client.atomic.directives.list_by_type("rule")
        assert len(rules) == 1

    def test_count_by_status(self, learn_client, clean_tables):
        """Test counting directives by status."""
        learn_client.atomic.directives.record(text="A")
        fact_id = learn_client.atomic.directives.record(text="B")
        learn_client.atomic.directives.record(text="C")

        learn_client.atomic.directives.set_status(fact_id, "paused")

        # count_by_status is not implemented, we'll verify counts via list_* methods
        active = learn_client.atomic.directives.list_active()
        assert len(active) == 2

        paused = learn_client.atomic.directives.list_by_type("standing", include_inactive=True)
        paused_count = len([d for d in paused if d.directive_details.status == "paused"])
        assert paused_count == 1
