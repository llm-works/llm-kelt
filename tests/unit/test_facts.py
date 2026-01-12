"""Tests for facts collection."""

import pytest

from llm_learn import ValidationError
from llm_learn.core.exceptions import NotFoundError


class TestFactsClient:
    """Test FactsClient functionality."""

    def test_add_fact(self, learn_client, clean_tables):
        """Test adding a simple fact."""
        fact_id = learn_client.facts.add("Response format: markdown")

        assert fact_id > 0

        fact = learn_client.facts.get(fact_id)
        assert fact is not None
        assert fact.content == "Response format: markdown"
        assert fact.source == "user"  # Default
        assert fact.confidence == 1.0  # Default
        assert fact.active is True  # Default

    def test_add_fact_with_category(self, learn_client, clean_tables):
        """Test adding a fact with category."""
        fact_id = learn_client.facts.add(
            "Timezone: UTC",
            category="settings",
        )

        fact = learn_client.facts.get(fact_id)
        assert fact.category == "settings"

    def test_add_fact_with_source(self, learn_client, clean_tables):
        """Test adding a fact with different source."""
        fact_id = learn_client.facts.add(
            "Preferred language: Python",
            source="inferred",
            confidence=0.8,
        )

        fact = learn_client.facts.get(fact_id)
        assert fact.source == "inferred"
        assert fact.confidence == 0.8

    def test_add_fact_empty_content_raises(self, learn_client, clean_tables):
        """Test that empty content raises ValidationError."""
        with pytest.raises(ValidationError, match="cannot be empty"):
            learn_client.facts.add("")

        with pytest.raises(ValidationError, match="cannot be empty"):
            learn_client.facts.add("   ")

    def test_add_fact_invalid_confidence_raises(self, learn_client, clean_tables):
        """Test that invalid confidence raises ValidationError."""
        with pytest.raises(ValidationError, match="Confidence must be"):
            learn_client.facts.add("Test", confidence=-0.1)

        with pytest.raises(ValidationError, match="Confidence must be"):
            learn_client.facts.add("Test", confidence=1.5)

    def test_update_fact(self, learn_client, clean_tables):
        """Test updating a fact."""
        fact_id = learn_client.facts.add("Original content", category="preferences")

        updated = learn_client.facts.update(
            fact_id,
            content="Updated content",
            category="settings",
            confidence=0.9,
        )

        assert updated.content == "Updated content"
        assert updated.category == "settings"
        assert updated.confidence == 0.9

    def test_update_nonexistent_fact_raises(self, learn_client, clean_tables):
        """Test updating non-existent fact raises NotFoundError."""
        with pytest.raises(NotFoundError):
            learn_client.facts.update(99999, content="New content")

    def test_deactivate_fact(self, learn_client, clean_tables):
        """Test deactivating a fact (soft delete)."""
        fact_id = learn_client.facts.add("Temporary fact")

        result = learn_client.facts.deactivate(fact_id)
        assert result is True

        fact = learn_client.facts.get(fact_id)
        assert fact.active is False

    def test_activate_fact(self, learn_client, clean_tables):
        """Test reactivating a deactivated fact."""
        fact_id = learn_client.facts.add("Temporary fact")
        learn_client.facts.deactivate(fact_id)

        result = learn_client.facts.activate(fact_id)
        assert result is True

        fact = learn_client.facts.get(fact_id)
        assert fact.active is True

    def test_list_active(self, learn_client, clean_tables):
        """Test listing active facts."""
        learn_client.facts.add("Fact 1")
        learn_client.facts.add("Fact 2")
        fact3_id = learn_client.facts.add("Fact 3")
        learn_client.facts.deactivate(fact3_id)

        active = learn_client.facts.list_active()
        assert len(active) == 2
        assert all(f.active for f in active)

    def test_list_active_by_category(self, learn_client, clean_tables):
        """Test listing active facts filtered by category."""
        learn_client.facts.add("Setting A", category="preferences")
        learn_client.facts.add("Setting B", category="preferences")
        learn_client.facts.add("Config X", category="settings")

        prefs = learn_client.facts.list_active(category="preferences")
        assert len(prefs) == 2
        assert all(f.category == "preferences" for f in prefs)

    def test_list_active_by_min_confidence(self, learn_client, clean_tables):
        """Test listing facts filtered by minimum confidence."""
        learn_client.facts.add("High conf", confidence=0.9)
        learn_client.facts.add("Low conf", confidence=0.3)

        high_conf = learn_client.facts.list_active(min_confidence=0.5)
        assert len(high_conf) == 1
        assert high_conf[0].content == "High conf"

    def test_list_by_category(self, learn_client, clean_tables):
        """Test listing all facts by category."""
        learn_client.facts.add("Rule 1", category="rules")
        fact2_id = learn_client.facts.add("Rule 2", category="rules")
        learn_client.facts.deactivate(fact2_id)

        # Active only (default)
        rules = learn_client.facts.list_by_category("rules")
        assert len(rules) == 1

        # Include inactive
        all_rules = learn_client.facts.list_by_category("rules", include_inactive=True)
        assert len(all_rules) == 2

    def test_list_by_source(self, learn_client, clean_tables):
        """Test listing facts by source."""
        learn_client.facts.add("Manual fact", source="user")
        learn_client.facts.add("Inferred fact", source="inferred")

        user_facts = learn_client.facts.list_by_source("user")
        assert len(user_facts) == 1
        assert user_facts[0].source == "user"

    def test_count_by_category(self, learn_client, clean_tables):
        """Test counting facts by category."""
        learn_client.facts.add("P1", category="preferences")
        learn_client.facts.add("P2", category="preferences")
        learn_client.facts.add("S1", category="settings")
        learn_client.facts.add("No category")

        counts = learn_client.facts.count_by_category()
        assert counts["preferences"] == 2
        assert counts["settings"] == 1
        assert counts["uncategorized"] == 1

    def test_search(self, learn_client, clean_tables):
        """Test searching facts by content."""
        learn_client.facts.add("Output format: JSON")
        learn_client.facts.add("Temperature unit: celsius")
        learn_client.facts.add("Language: English")

        results = learn_client.facts.search("format")
        assert len(results) == 1
        assert "format" in results[0].content.lower()

        results = learn_client.facts.search("celsius")
        assert len(results) == 1

    def test_count(self, learn_client, clean_tables):
        """Test counting all facts."""
        assert learn_client.facts.count() == 0

        learn_client.facts.add("Fact 1")
        learn_client.facts.add("Fact 2")

        assert learn_client.facts.count() == 2

    def test_delete(self, learn_client, clean_tables):
        """Test hard deleting a fact."""
        fact_id = learn_client.facts.add("To be deleted")

        assert learn_client.facts.exists(fact_id)

        result = learn_client.facts.delete(fact_id)
        assert result is True

        assert not learn_client.facts.exists(fact_id)
