"""Tests for preference pairs collection."""

import pytest

from llm_learn import ValidationError


class TestPreferencesClient:
    """Test PreferencesClient functionality."""

    def test_record_preference_pair(self, learn_client, clean_tables):
        """Test recording a preference pair."""
        fact_id = learn_client.atomic.preferences.record(
            context="Summarize this article",
            chosen="Concise 3-bullet summary",
            rejected="Verbose 500-word essay",
            margin=0.7,
            category="synthesis",
        )

        assert fact_id > 0

        fact = learn_client.atomic.preferences.get(fact_id)
        assert fact is not None
        assert fact.preference_details.context == "Summarize this article"
        assert fact.preference_details.chosen == "Concise 3-bullet summary"
        assert fact.preference_details.rejected == "Verbose 500-word essay"
        assert fact.preference_details.margin == 0.7
        assert fact.category == "synthesis"

    def test_record_without_margin(self, learn_client, clean_tables):
        """Test recording pair without margin."""
        fact_id = learn_client.atomic.preferences.record(
            context="Context",
            chosen="Good response",
            rejected="Bad response",
        )

        fact = learn_client.atomic.preferences.get(fact_id)
        assert fact.preference_details.margin is None

    def test_record_with_metadata(self, learn_client, clean_tables):
        """Test recording pair with metadata."""
        fact_id = learn_client.atomic.preferences.record(
            context="Context",
            chosen="Good",
            rejected="Bad",
            metadata={"model": "gpt-4", "temperature": 0.7},
        )

        fact = learn_client.atomic.preferences.get(fact_id)
        assert fact.preference_details.metadata_ == {"model": "gpt-4", "temperature": 0.7}

    def test_empty_context_raises(self, learn_client, clean_tables):
        """Test that empty context raises ValidationError."""
        with pytest.raises(ValidationError, match="context cannot be empty"):
            learn_client.atomic.preferences.record(
                context="",
                chosen="Good",
                rejected="Bad",
            )

    def test_empty_chosen_raises(self, learn_client, clean_tables):
        """Test that empty chosen raises ValidationError."""
        with pytest.raises(ValidationError, match="chosen cannot be empty"):
            learn_client.atomic.preferences.record(
                context="Context",
                chosen="",
                rejected="Bad",
            )

    def test_empty_rejected_raises(self, learn_client, clean_tables):
        """Test that empty rejected raises ValidationError."""
        with pytest.raises(ValidationError, match="rejected cannot be empty"):
            learn_client.atomic.preferences.record(
                context="Context",
                chosen="Good",
                rejected="",
            )

    def test_invalid_margin_raises(self, learn_client, clean_tables):
        """Test that invalid margin raises ValidationError."""
        with pytest.raises(ValidationError, match="margin must be"):
            learn_client.atomic.preferences.record(
                context="Context",
                chosen="Good",
                rejected="Bad",
                margin=1.5,
            )

    def test_list_by_category(self, learn_client, clean_tables):
        """Test listing pairs by category."""
        learn_client.atomic.preferences.record(context="A", chosen="G", rejected="B", category="synthesis")
        learn_client.atomic.preferences.record(context="B", chosen="G", rejected="B", category="analysis")
        learn_client.atomic.preferences.record(context="C", chosen="G", rejected="B", category="synthesis")

        synthesis = learn_client.atomic.preferences.list_by_category("synthesis")
        assert len(synthesis) == 2

        analysis = learn_client.atomic.preferences.list_by_category("analysis")
        assert len(analysis) == 1

    def test_get_categories(self, learn_client, clean_tables):
        """Test getting unique categories."""
        learn_client.atomic.preferences.record(context="A", chosen="G", rejected="B", category="synthesis")
        learn_client.atomic.preferences.record(context="B", chosen="G", rejected="B", category="analysis")
        learn_client.atomic.preferences.record(context="C", chosen="G", rejected="B", category="synthesis")

        categories = learn_client.atomic.preferences.get_categories()
        assert set(categories) == {"analysis", "synthesis"}

    def test_count(self, learn_client, clean_tables):
        """Test counting preference pairs."""
        assert learn_client.atomic.preferences.count() == 0

        learn_client.atomic.preferences.record(context="A", chosen="G", rejected="B")
        learn_client.atomic.preferences.record(context="B", chosen="G", rejected="B")

        assert learn_client.atomic.preferences.count() == 2
