"""Tests for directives collection."""

from datetime import UTC, datetime, timedelta

import pytest

from llm_learn import NotFoundError, ValidationError


class TestDirectivesClient:
    """Test DirectivesClient functionality."""

    def test_record_directive(self, learn_client, clean_tables):
        """Test recording a directive."""
        directive_id = learn_client.directives.record(
            text="Focus more on AI safety research",
            directive_type="standing",
        )

        assert directive_id > 0

        directive = learn_client.directives.get(directive_id)
        assert directive is not None
        assert directive.directive_text == "Focus more on AI safety research"
        assert directive.directive_type == "standing"
        assert directive.status == "active"

    def test_record_one_time(self, learn_client, clean_tables):
        """Test recording a one-time directive."""
        directive_id = learn_client.directives.record(
            text="Research vector databases",
            directive_type="one-time",
        )

        directive = learn_client.directives.get(directive_id)
        assert directive.directive_type == "one-time"

    def test_record_rule(self, learn_client, clean_tables):
        """Test recording a rule directive."""
        directive_id = learn_client.directives.record(
            text="Never show crypto price news",
            directive_type="rule",
            parsed_rules={"action": "exclude", "topic": "crypto_prices"},
        )

        directive = learn_client.directives.get(directive_id)
        assert directive.directive_type == "rule"
        assert directive.parsed_rules["action"] == "exclude"

    def test_record_with_expiration(self, learn_client, clean_tables):
        """Test recording directive with expiration."""
        expires = datetime.now(UTC) + timedelta(days=30)

        directive_id = learn_client.directives.record(
            text="Temporary directive",
            expires_at=expires,
        )

        directive = learn_client.directives.get(directive_id)
        assert directive.expires_at is not None

    def test_empty_text_raises(self, learn_client, clean_tables):
        """Test that empty text raises ValidationError."""
        with pytest.raises(ValidationError, match="Directive text cannot be empty"):
            learn_client.directives.record(text="")

    def test_invalid_type_raises(self, learn_client, clean_tables):
        """Test that invalid type raises ValidationError."""
        with pytest.raises(ValidationError, match="Invalid directive type"):
            learn_client.directives.record(
                text="Test",
                directive_type="invalid",
            )

    def test_set_status(self, learn_client, clean_tables):
        """Test updating directive status."""
        directive_id = learn_client.directives.record(text="Test directive")

        # Pause
        result = learn_client.directives.set_status(directive_id, "paused")
        assert result is True

        directive = learn_client.directives.get(directive_id)
        assert directive.status == "paused"

        # Complete
        learn_client.directives.set_status(directive_id, "completed")
        directive = learn_client.directives.get(directive_id)
        assert directive.status == "completed"

    def test_set_status_not_found_raises(self, learn_client, clean_tables):
        """Test that setting status on non-existent directive raises NotFoundError."""
        with pytest.raises(NotFoundError):
            learn_client.directives.set_status(99999, "paused")

    def test_set_status_invalid_raises(self, learn_client, clean_tables):
        """Test that invalid status raises ValidationError."""
        directive_id = learn_client.directives.record(text="Test")

        with pytest.raises(ValidationError, match="Invalid status"):
            learn_client.directives.set_status(directive_id, "invalid")

    def test_list_active(self, learn_client, clean_tables):
        """Test listing active directives."""
        learn_client.directives.record(text="Active 1")
        directive_id = learn_client.directives.record(text="To be paused")
        learn_client.directives.record(text="Active 2")

        # Pause one
        learn_client.directives.set_status(directive_id, "paused")

        active = learn_client.directives.list_active()
        assert len(active) == 2

    def test_list_active_filters_expired(self, learn_client, clean_tables):
        """Test that list_active filters out expired directives."""
        past = datetime.now(UTC) - timedelta(days=1)
        future = datetime.now(UTC) + timedelta(days=1)

        learn_client.directives.record(text="Expired", expires_at=past)
        learn_client.directives.record(text="Not expired", expires_at=future)
        learn_client.directives.record(text="No expiration")

        active = learn_client.directives.list_active()
        assert len(active) == 2
        texts = [d.directive_text for d in active]
        assert "Expired" not in texts

    def test_list_by_type(self, learn_client, clean_tables):
        """Test listing directives by type."""
        learn_client.directives.record(text="A", directive_type="standing")
        learn_client.directives.record(text="B", directive_type="rule")
        learn_client.directives.record(text="C", directive_type="standing")

        standing = learn_client.directives.list_by_type("standing")
        assert len(standing) == 2

        rules = learn_client.directives.list_by_type("rule")
        assert len(rules) == 1

    def test_count_by_status(self, learn_client, clean_tables):
        """Test counting directives by status."""
        learn_client.directives.record(text="A")
        directive_id = learn_client.directives.record(text="B")
        learn_client.directives.record(text="C")

        learn_client.directives.set_status(directive_id, "paused")

        counts = learn_client.directives.count_by_status()
        assert counts["active"] == 2
        assert counts["paused"] == 1
        assert counts["completed"] == 0
