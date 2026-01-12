"""Tests for preference pairs collection."""

import pytest

from llm_learn import ValidationError


class TestPreferencesClient:
    """Test PreferencesClient functionality."""

    def test_record_preference_pair(self, learn_client, clean_tables):
        """Test recording a preference pair."""
        pair_id = learn_client.preferences.record(
            context="Summarize this article",
            chosen="Concise 3-bullet summary",
            rejected="Verbose 500-word essay",
            margin=0.7,
            domain="synthesis",
        )

        assert pair_id > 0

        pair = learn_client.preferences.get(pair_id)
        assert pair is not None
        assert pair.context == "Summarize this article"
        assert pair.chosen == "Concise 3-bullet summary"
        assert pair.rejected == "Verbose 500-word essay"
        assert pair.margin == 0.7
        assert pair.domain == "synthesis"

    def test_record_without_margin(self, learn_client, clean_tables):
        """Test recording pair without margin."""
        pair_id = learn_client.preferences.record(
            context="Context",
            chosen="Good response",
            rejected="Bad response",
        )

        pair = learn_client.preferences.get(pair_id)
        assert pair.margin is None

    def test_record_with_metadata(self, learn_client, clean_tables):
        """Test recording pair with metadata."""
        pair_id = learn_client.preferences.record(
            context="Context",
            chosen="Good",
            rejected="Bad",
            metadata={"model": "gpt-4", "temperature": 0.7},
        )

        pair = learn_client.preferences.get(pair_id)
        assert pair.metadata_ == {"model": "gpt-4", "temperature": 0.7}

    def test_empty_context_raises(self, learn_client, clean_tables):
        """Test that empty context raises ValidationError."""
        with pytest.raises(ValidationError, match="Context cannot be empty"):
            learn_client.preferences.record(
                context="",
                chosen="Good",
                rejected="Bad",
            )

    def test_empty_chosen_raises(self, learn_client, clean_tables):
        """Test that empty chosen raises ValidationError."""
        with pytest.raises(ValidationError, match="Chosen response cannot be empty"):
            learn_client.preferences.record(
                context="Context",
                chosen="",
                rejected="Bad",
            )

    def test_empty_rejected_raises(self, learn_client, clean_tables):
        """Test that empty rejected raises ValidationError."""
        with pytest.raises(ValidationError, match="Rejected response cannot be empty"):
            learn_client.preferences.record(
                context="Context",
                chosen="Good",
                rejected="",
            )

    def test_invalid_margin_raises(self, learn_client, clean_tables):
        """Test that invalid margin raises ValidationError."""
        with pytest.raises(ValidationError, match="Margin must be"):
            learn_client.preferences.record(
                context="Context",
                chosen="Good",
                rejected="Bad",
                margin=1.5,
            )

    def test_list_by_domain(self, learn_client, clean_tables):
        """Test listing pairs by domain."""
        learn_client.preferences.record(context="A", chosen="G", rejected="B", domain="synthesis")
        learn_client.preferences.record(context="B", chosen="G", rejected="B", domain="analysis")
        learn_client.preferences.record(context="C", chosen="G", rejected="B", domain="synthesis")

        synthesis = learn_client.preferences.list_by_domain("synthesis")
        assert len(synthesis) == 2

        analysis = learn_client.preferences.list_by_domain("analysis")
        assert len(analysis) == 1

    def test_list_domains(self, learn_client, clean_tables):
        """Test listing unique domains."""
        learn_client.preferences.record(context="A", chosen="G", rejected="B", domain="synthesis")
        learn_client.preferences.record(context="B", chosen="G", rejected="B", domain="analysis")
        learn_client.preferences.record(context="C", chosen="G", rejected="B", domain="synthesis")

        domains = learn_client.preferences.list_domains()
        assert set(domains) == {"analysis", "synthesis"}

    def test_count(self, learn_client, clean_tables):
        """Test counting preference pairs."""
        assert learn_client.preferences.count() == 0

        learn_client.preferences.record(context="A", chosen="G", rejected="B")
        learn_client.preferences.record(context="B", chosen="G", rejected="B")

        assert learn_client.preferences.count() == 2
