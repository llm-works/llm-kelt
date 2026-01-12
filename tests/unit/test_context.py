"""Tests for context builder (fact injection into system prompts)."""

import pytest

from llm_learn.inference.context import ContextBuilder


class TestContextBuilder:
    """Test ContextBuilder functionality."""

    def test_build_empty_prompt_no_facts(self, learn_client, clean_tables):
        """Test building prompt with no facts returns base prompt."""
        builder = ContextBuilder(learn_client.facts)

        result = builder.build_system_prompt("You are a helpful assistant.")
        assert result == "You are a helpful assistant."

    def test_build_prompt_with_facts(self, learn_client, clean_tables):
        """Test building prompt with facts appends them."""
        learn_client.facts.add("Response format: markdown")
        learn_client.facts.add("Timezone: UTC")

        builder = ContextBuilder(learn_client.facts)
        result = builder.build_system_prompt("You are a helpful assistant.")

        assert "You are a helpful assistant." in result
        assert "## About the user:" in result
        assert "Response format: markdown" in result
        assert "Timezone: UTC" in result

    def test_build_prompt_facts_prepend(self, learn_client, clean_tables):
        """Test prepending facts to system prompt."""
        learn_client.facts.add("Output language: English")

        builder = ContextBuilder(learn_client.facts)
        result = builder.build_system_prompt(
            "You are a helpful assistant.",
            fact_position="prepend",
        )

        # Facts should come before base prompt
        facts_pos = result.find("## About the user:")
        base_pos = result.find("You are a helpful assistant.")
        assert facts_pos < base_pos

    def test_build_prompt_facts_append(self, learn_client, clean_tables):
        """Test appending facts to system prompt (default)."""
        learn_client.facts.add("Output language: English")

        builder = ContextBuilder(learn_client.facts)
        result = builder.build_system_prompt("You are a helpful assistant.")

        # Facts should come after base prompt
        facts_pos = result.find("## About the user:")
        base_pos = result.find("You are a helpful assistant.")
        assert base_pos < facts_pos

    def test_build_prompt_grouped_by_category(self, learn_client, clean_tables):
        """Test facts are grouped by category."""
        learn_client.facts.add("Code style: verbose", category="preferences")
        learn_client.facts.add("Include examples: yes", category="preferences")
        learn_client.facts.add("Domain: backend systems", category="context")

        builder = ContextBuilder(learn_client.facts)
        result = builder.build_system_prompt("")

        assert "### Preferences" in result
        assert "### Context" in result

    def test_build_prompt_uncategorized_no_headers(self, learn_client, clean_tables):
        """Test uncategorized facts don't get category headers."""
        learn_client.facts.add("Fact without category")
        learn_client.facts.add("Another uncategorized fact")

        builder = ContextBuilder(learn_client.facts)
        result = builder.build_system_prompt("")

        assert "## About the user:" in result
        assert "###" not in result  # No category headers

    def test_build_prompt_filter_by_category(self, learn_client, clean_tables):
        """Test filtering facts by category."""
        learn_client.facts.add("Setting A", category="preferences")
        learn_client.facts.add("Config B", category="settings")

        builder = ContextBuilder(learn_client.facts)
        result = builder.build_system_prompt("", categories=["preferences"])

        assert "Setting A" in result
        assert "Config B" not in result

    def test_build_prompt_filter_by_min_confidence(self, learn_client, clean_tables):
        """Test filtering facts by minimum confidence."""
        learn_client.facts.add("High confidence fact", confidence=0.9)
        learn_client.facts.add("Low confidence fact", confidence=0.3)

        builder = ContextBuilder(learn_client.facts)
        result = builder.build_system_prompt("", min_confidence=0.5)

        assert "High confidence fact" in result
        assert "Low confidence fact" not in result

    def test_build_prompt_excludes_inactive(self, learn_client, clean_tables):
        """Test that inactive facts are excluded."""
        learn_client.facts.add("Active fact")
        inactive_id = learn_client.facts.add("Inactive fact")
        learn_client.facts.deactivate(inactive_id)

        builder = ContextBuilder(learn_client.facts)
        result = builder.build_system_prompt("")

        assert "Active fact" in result
        assert "Inactive fact" not in result

    def test_build_prompt_max_facts(self, learn_client, clean_tables):
        """Test limiting number of facts."""
        for i in range(10):
            learn_client.facts.add(f"Fact {i}")

        builder = ContextBuilder(learn_client.facts)
        result = builder.build_system_prompt("", max_facts=3)

        # Count how many "Fact" occurrences (should be 3)
        fact_count = result.count("- Fact")
        assert fact_count == 3

    def test_get_facts_summary(self, learn_client, clean_tables):
        """Test getting facts summary."""
        learn_client.facts.add("Pref 1", category="preferences")
        learn_client.facts.add("Pref 2", category="preferences")
        learn_client.facts.add("Setting X", category="settings", confidence=0.8)

        builder = ContextBuilder(learn_client.facts)
        summary = builder.get_facts_summary()

        assert summary["total_active"] == 3
        assert summary["by_category"]["preferences"] == 2
        assert summary["by_category"]["settings"] == 1
        assert len(summary["facts"]) == 3

    def test_build_empty_base_prompt_with_facts(self, learn_client, clean_tables):
        """Test building with empty base prompt still returns facts."""
        learn_client.facts.add("Some fact")

        builder = ContextBuilder(learn_client.facts)
        result = builder.build_system_prompt("")

        assert "## About the user:" in result
        assert "Some fact" in result
