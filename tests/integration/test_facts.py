"""Tests for assertions collection (formerly facts)."""

import pytest

from llm_learn import ValidationError


class TestAssertionsClient:
    """Test AssertionsClient functionality."""

    def test_add_assertion(self, learn_client, clean_tables):
        """Test adding a simple assertion."""
        fact_id = learn_client.atomic.assertions.add("Response format: markdown")

        assert fact_id > 0

        fact = learn_client.atomic.assertions.get(fact_id)
        assert fact is not None
        assert fact.content == "Response format: markdown"
        assert fact.source == "user"  # Default
        assert fact.confidence == 1.0  # Default
        assert fact.active is True  # Default

    def test_add_assertion_with_category(self, learn_client, clean_tables):
        """Test adding an assertion with category."""
        fact_id = learn_client.atomic.assertions.add(
            "Timezone: UTC",
            category="settings",
        )

        fact = learn_client.atomic.assertions.get(fact_id)
        assert fact.category == "settings"

    def test_add_assertion_with_source(self, learn_client, clean_tables):
        """Test adding an assertion with different source."""
        fact_id = learn_client.atomic.assertions.add(
            "Preferred language: Python",
            source="inferred",
            confidence=0.8,
        )

        fact = learn_client.atomic.assertions.get(fact_id)
        assert fact.source == "inferred"
        assert fact.confidence == 0.8

    def test_add_assertion_empty_content_raises(self, learn_client, clean_tables):
        """Test that empty content raises ValidationError."""
        with pytest.raises(ValidationError, match="cannot be empty"):
            learn_client.atomic.assertions.add("")

        with pytest.raises(ValidationError, match="cannot be empty"):
            learn_client.atomic.assertions.add("   ")

    def test_add_assertion_invalid_confidence_raises(self, learn_client, clean_tables):
        """Test that invalid confidence raises ValidationError."""
        with pytest.raises(ValidationError, match="confidence must be"):
            learn_client.atomic.assertions.add("Test", confidence=-0.1)

        with pytest.raises(ValidationError, match="confidence must be"):
            learn_client.atomic.assertions.add("Test", confidence=1.5)

    def test_update_assertion(self, learn_client, clean_tables):
        """Test updating an assertion."""
        fact_id = learn_client.atomic.assertions.add("Original content", category="preferences")

        updated = learn_client.atomic.assertions.update(
            fact_id,
            content="Updated content",
            category="settings",
            confidence=0.9,
        )

        assert updated is True

        fact = learn_client.atomic.assertions.get(fact_id)
        assert fact.content == "Updated content"
        assert fact.category == "settings"
        assert fact.confidence == 0.9

    def test_update_nonexistent_raises(self, learn_client, clean_tables):
        """Test updating non-existent assertion returns False."""
        updated = learn_client.atomic.assertions.update(99999, content="New content")
        assert updated is False

    def test_deactivate_assertion(self, learn_client, clean_tables):
        """Test deactivating an assertion (soft delete)."""
        fact_id = learn_client.atomic.assertions.add("Temporary assertion")

        result = learn_client.atomic.assertions.deactivate(fact_id)
        assert result is True

        fact = learn_client.atomic.assertions.get(fact_id)
        assert fact.active is False

    def test_activate_assertion(self, learn_client, clean_tables):
        """Test reactivating a deactivated assertion."""
        fact_id = learn_client.atomic.assertions.add("Temporary assertion")
        learn_client.atomic.assertions.deactivate(fact_id)

        result = learn_client.atomic.assertions.activate(fact_id)
        assert result is True

        fact = learn_client.atomic.assertions.get(fact_id)
        assert fact.active is True

    def test_list_active(self, learn_client, clean_tables):
        """Test listing active assertions."""
        learn_client.atomic.assertions.add("Assertion 1")
        learn_client.atomic.assertions.add("Assertion 2")
        fact3_id = learn_client.atomic.assertions.add("Assertion 3")
        learn_client.atomic.assertions.deactivate(fact3_id)

        active = learn_client.atomic.assertions.list_active()
        assert len(active) == 2
        assert all(f.active for f in active)

    def test_list_active_by_category(self, learn_client, clean_tables):
        """Test listing active assertions filtered by category."""
        learn_client.atomic.assertions.add("Setting A", category="preferences")
        learn_client.atomic.assertions.add("Setting B", category="preferences")
        learn_client.atomic.assertions.add("Config X", category="settings")

        prefs = learn_client.atomic.assertions.list_active(category="preferences")
        assert len(prefs) == 2
        assert all(f.category == "preferences" for f in prefs)

    def test_list_active_by_min_confidence(self, learn_client, clean_tables):
        """Test listing assertions filtered by minimum confidence."""
        learn_client.atomic.assertions.add("High conf", confidence=0.9)
        learn_client.atomic.assertions.add("Low conf", confidence=0.3)

        high_conf = learn_client.atomic.assertions.list_active(min_confidence=0.5)
        assert len(high_conf) == 1
        assert high_conf[0].content == "High conf"

    def test_list_by_category(self, learn_client, clean_tables):
        """Test listing all assertions by category."""
        learn_client.atomic.assertions.add("Rule 1", category="rules")
        fact2_id = learn_client.atomic.assertions.add("Rule 2", category="rules")
        learn_client.atomic.assertions.deactivate(fact2_id)

        # Active only (default)
        rules = learn_client.atomic.assertions.list_by_category("rules")
        assert len(rules) == 1

        # Include inactive
        all_rules = learn_client.atomic.assertions.list_by_category("rules", active_only=False)
        assert len(all_rules) == 2

    def test_list_by_source(self, learn_client, clean_tables):
        """Test listing assertions by source."""
        learn_client.atomic.assertions.add("Manual assertion", source="user")
        learn_client.atomic.assertions.add("Inferred assertion", source="inferred")

        user_assertions = learn_client.atomic.assertions.list_by_source("user")
        assert len(user_assertions) == 1
        assert user_assertions[0].source == "user"

    def test_count_by_category(self, learn_client, clean_tables):
        """Test counting assertions by category."""
        learn_client.atomic.assertions.add("P1", category="preferences")
        learn_client.atomic.assertions.add("P2", category="preferences")
        learn_client.atomic.assertions.add("S1", category="settings")
        learn_client.atomic.assertions.add("No category")

        counts = learn_client.atomic.assertions.count_by_category()
        assert counts["preferences"] == 2
        assert counts["settings"] == 1
        assert counts.get(None, 0) == 1  # None key for uncategorized

    def test_search(self, learn_client, clean_tables):
        """Test searching assertions by content."""
        learn_client.atomic.assertions.add("Output format: JSON")
        learn_client.atomic.assertions.add("Temperature unit: celsius")
        learn_client.atomic.assertions.add("Language: English")

        results = learn_client.atomic.assertions.search("format")
        assert len(results) == 1
        assert "format" in results[0].content.lower()

        results = learn_client.atomic.assertions.search("celsius")
        assert len(results) == 1

    def test_count(self, learn_client, clean_tables):
        """Test counting all assertions."""
        assert learn_client.atomic.assertions.count() == 0

        learn_client.atomic.assertions.add("Assertion 1")
        learn_client.atomic.assertions.add("Assertion 2")

        assert learn_client.atomic.assertions.count() == 2

    def test_delete(self, learn_client, clean_tables):
        """Test hard deleting an assertion."""
        fact_id = learn_client.atomic.assertions.add("To be deleted")

        assert learn_client.atomic.assertions.exists(fact_id)

        result = learn_client.atomic.assertions.delete(fact_id)
        assert result is True

        assert not learn_client.atomic.assertions.exists(fact_id)


class TestAssertionsEmbeddings:
    """Test embedding functionality via Protocol.embeddings (requires pgvector)."""

    def test_set_embedding(self, learn_client, clean_tables):
        """Test setting an embedding for an assertion."""
        fact_id = learn_client.atomic.assertions.add("User prefers Python")
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]

        # Should not raise - use embeddings adapter
        learn_client.atomic.embeddings.set_embedding(fact_id, embedding, "test-model")

        # Verify by listing assertions without embeddings
        without = learn_client.atomic.embeddings.list_without_embeddings("test-model")
        assert all(f.id != fact_id for f in without)

    def test_set_embedding_upsert(self, learn_client, clean_tables):
        """Test that set_embedding updates existing embedding."""
        fact_id = learn_client.atomic.assertions.add("Test assertion")
        embedding1 = [0.1, 0.2, 0.3]
        embedding2 = [0.4, 0.5, 0.6]

        learn_client.atomic.embeddings.set_embedding(fact_id, embedding1, "test-model")
        learn_client.atomic.embeddings.set_embedding(fact_id, embedding2, "test-model")

        # Should still only have one embedding for this model
        # Verify via search_similar - should find the assertion with new embedding
        results = learn_client.atomic.embeddings.search_similar(
            query=[0.4, 0.5, 0.6],
            model_name="test-model",
            min_similarity=0.9,
        )
        assert len(results) == 1
        assert results[0].entity.id == fact_id

    def test_set_embedding_different_models(self, learn_client, clean_tables):
        """Test setting embeddings with different model names."""
        fact_id = learn_client.atomic.assertions.add("Multi-model assertion")
        embedding1 = [0.1, 0.2, 0.3]
        embedding2 = [0.4, 0.5, 0.6, 0.7]  # Different dimensions

        learn_client.atomic.embeddings.set_embedding(fact_id, embedding1, "model-a")
        learn_client.atomic.embeddings.set_embedding(fact_id, embedding2, "model-b")

        # Assertion should not appear in list_without_embeddings for either model
        without_a = learn_client.atomic.embeddings.list_without_embeddings("model-a")
        without_b = learn_client.atomic.embeddings.list_without_embeddings("model-b")
        assert all(f.id != fact_id for f in without_a)
        assert all(f.id != fact_id for f in without_b)

        # But should appear for a different model
        without_c = learn_client.atomic.embeddings.list_without_embeddings("model-c")
        assert any(f.id == fact_id for f in without_c)

    def test_list_without_embeddings(self, learn_client, clean_tables):
        """Test listing assertions without embeddings for a model."""
        fact1_id = learn_client.atomic.assertions.add("Assertion with embedding")
        fact2_id = learn_client.atomic.assertions.add("Assertion without embedding")
        fact3_id = learn_client.atomic.assertions.add("Another without")

        learn_client.atomic.embeddings.set_embedding(fact1_id, [0.1, 0.2, 0.3], "test-model")

        without = learn_client.atomic.embeddings.list_without_embeddings("test-model")

        fact_ids = [f.id for f in without]
        assert fact1_id not in fact_ids
        assert fact2_id in fact_ids
        assert fact3_id in fact_ids

    def test_list_without_embeddings_excludes_inactive(self, learn_client, clean_tables):
        """Test that list_without_embeddings excludes inactive assertions."""
        fact1_id = learn_client.atomic.assertions.add("Active assertion")
        fact2_id = learn_client.atomic.assertions.add("Inactive assertion")
        learn_client.atomic.assertions.deactivate(fact2_id)

        without = learn_client.atomic.embeddings.list_without_embeddings("test-model")

        fact_ids = [f.id for f in without]
        assert fact1_id in fact_ids
        assert fact2_id not in fact_ids

    def test_list_without_embeddings_respects_limit(self, learn_client, clean_tables):
        """Test list_without_embeddings respects limit parameter."""
        for i in range(10):
            learn_client.atomic.assertions.add(f"Assertion {i}")

        without = learn_client.atomic.embeddings.list_without_embeddings("test-model", limit=3)
        assert len(without) == 3

    def test_search_similar(self, learn_client, clean_tables):
        """Test searching assertions by embedding similarity."""
        fact1_id = learn_client.atomic.assertions.add("Python programming")
        fact2_id = learn_client.atomic.assertions.add("JavaScript development")
        fact3_id = learn_client.atomic.assertions.add("Database design")

        # Set embeddings - fact1 and fact2 similar, fact3 different
        learn_client.atomic.embeddings.set_embedding(fact1_id, [0.9, 0.1, 0.0], "test")
        learn_client.atomic.embeddings.set_embedding(fact2_id, [0.8, 0.2, 0.0], "test")
        learn_client.atomic.embeddings.set_embedding(fact3_id, [0.0, 0.0, 1.0], "test")

        # Search for something similar to fact1
        results = learn_client.atomic.embeddings.search_similar(
            query=[0.85, 0.15, 0.0],
            model_name="test",
            top_k=10,
            min_similarity=0.5,
        )

        # Should find fact1 and fact2, not fact3
        assert len(results) >= 2
        result_ids = [r.entity.id for r in results]
        assert fact1_id in result_ids
        assert fact2_id in result_ids
        # fact3 might not be in results due to low similarity

        # Results should be sorted by similarity (highest first)
        similarities = [r.score for r in results]
        assert similarities == sorted(similarities, reverse=True)

    def test_search_similar_min_similarity_filter(self, learn_client, clean_tables):
        """Test that search_similar respects min_similarity threshold."""
        fact1_id = learn_client.atomic.assertions.add("Very similar")
        fact2_id = learn_client.atomic.assertions.add("Less similar")

        learn_client.atomic.embeddings.set_embedding(fact1_id, [1.0, 0.0, 0.0], "test")
        learn_client.atomic.embeddings.set_embedding(fact2_id, [0.5, 0.5, 0.5], "test")

        # High threshold - should only match very similar
        results = learn_client.atomic.embeddings.search_similar(
            query=[1.0, 0.0, 0.0],
            model_name="test",
            min_similarity=0.95,
        )

        result_ids = [r.entity.id for r in results]
        assert fact1_id in result_ids
        # fact2 should be filtered out due to low similarity

    def test_search_similar_top_k_limit(self, learn_client, clean_tables):
        """Test that search_similar respects top_k limit."""
        for i in range(10):
            fact_id = learn_client.atomic.assertions.add(f"Assertion {i}")
            learn_client.atomic.embeddings.set_embedding(fact_id, [0.5, 0.5, 0.0], "test")

        results = learn_client.atomic.embeddings.search_similar(
            query=[0.5, 0.5, 0.0],
            model_name="test",
            top_k=3,
            min_similarity=0.0,
        )

        assert len(results) == 3

    def test_search_similar_active_only(self, learn_client, clean_tables):
        """Test that search_similar excludes inactive assertions by default."""
        fact1_id = learn_client.atomic.assertions.add("Active assertion")
        fact2_id = learn_client.atomic.assertions.add("Inactive assertion")

        learn_client.atomic.embeddings.set_embedding(fact1_id, [1.0, 0.0], "test")
        learn_client.atomic.embeddings.set_embedding(fact2_id, [1.0, 0.0], "test")
        learn_client.atomic.assertions.deactivate(fact2_id)

        # Default: active assertions only
        results = learn_client.atomic.embeddings.search_similar(
            query=[1.0, 0.0],
            model_name="test",
            min_similarity=0.0,
        )

        result_ids = [r.entity.id for r in results]
        assert fact1_id in result_ids
        assert fact2_id not in result_ids

    def test_search_similar_empty_results(self, learn_client, clean_tables):
        """Test search_similar returns empty list when no matches."""
        fact_id = learn_client.atomic.assertions.add("Test assertion")
        learn_client.atomic.embeddings.set_embedding(fact_id, [1.0, 0.0, 0.0], "test")

        # Search with completely different embedding and high threshold
        results = learn_client.atomic.embeddings.search_similar(
            query=[0.0, 0.0, 1.0],
            model_name="test",
            min_similarity=0.99,
        )

        assert results == []

    def test_search_similar_wrong_model_returns_empty(self, learn_client, clean_tables):
        """Test search_similar returns empty when model doesn't match."""
        fact_id = learn_client.atomic.assertions.add("Test assertion")
        learn_client.atomic.embeddings.set_embedding(fact_id, [1.0, 0.0], "model-a")

        results = learn_client.atomic.embeddings.search_similar(
            query=[1.0, 0.0],
            model_name="model-b",  # Different model
            min_similarity=0.0,
        )

        assert results == []

    def test_delete_embedding_on_fact_delete(self, learn_client, clean_tables, database):
        """Test that embeddings can be cleaned up when deleting facts."""
        fact_id = learn_client.atomic.assertions.add("Assertion to delete")
        learn_client.atomic.embeddings.set_embedding(fact_id, [0.1, 0.2], "test")

        # Verify embedding exists
        results = learn_client.atomic.embeddings.search_similar(
            query=[0.1, 0.2],
            model_name="test",
            min_similarity=0.9,
        )
        assert len(results) == 1

        # Delete embedding explicitly, then delete the assertion
        learn_client.atomic.embeddings.delete_embedding(fact_id)
        learn_client.atomic.assertions.delete(fact_id)

        # Embedding should be gone
        results_after = learn_client.atomic.embeddings.search_similar(
            query=[0.1, 0.2],
            model_name="test",
            min_similarity=0.0,
        )
        assert len(results_after) == 0

    def test_search_similar_with_categories_filter(self, learn_client, clean_tables):
        """Test that search_similar filters by categories in SQL."""
        # Create assertions in different categories with similar embeddings
        pref_id = learn_client.atomic.assertions.add("User prefers Python", category="preferences")
        rule_id = learn_client.atomic.assertions.add("Always use type hints", category="rules")
        bg_id = learn_client.atomic.assertions.add("10 years experience", category="background")

        # All assertions have similar embeddings
        learn_client.atomic.embeddings.set_embedding(pref_id, [0.9, 0.1, 0.0], "test")
        learn_client.atomic.embeddings.set_embedding(rule_id, [0.85, 0.15, 0.0], "test")
        learn_client.atomic.embeddings.set_embedding(bg_id, [0.8, 0.2, 0.0], "test")

        # Search without categories - should get all
        results_all = learn_client.atomic.embeddings.search_similar(
            query=[0.9, 0.1, 0.0],
            model_name="test",
            min_similarity=0.5,
        )
        assert len(results_all) == 3

        # Search with single category filter
        results_prefs = learn_client.atomic.embeddings.search_similar(
            query=[0.9, 0.1, 0.0],
            model_name="test",
            min_similarity=0.5,
            categories=["preferences"],
        )
        assert len(results_prefs) == 1
        assert results_prefs[0].entity.id == pref_id

        # Search with multiple categories
        results_multi = learn_client.atomic.embeddings.search_similar(
            query=[0.9, 0.1, 0.0],
            model_name="test",
            min_similarity=0.5,
            categories=["preferences", "rules"],
        )
        result_ids = [r.entity.id for r in results_multi]
        assert len(results_multi) == 2
        assert pref_id in result_ids
        assert rule_id in result_ids
        assert bg_id not in result_ids

        # Search with non-existent category
        results_empty = learn_client.atomic.embeddings.search_similar(
            query=[0.9, 0.1, 0.0],
            model_name="test",
            min_similarity=0.5,
            categories=["nonexistent"],
        )
        assert len(results_empty) == 0
