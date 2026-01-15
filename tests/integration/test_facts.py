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


class TestFactsEmbeddings:
    """Test FactsClient embedding functionality (requires pgvector)."""

    def test_set_embedding(self, learn_client, clean_tables):
        """Test setting an embedding for a fact."""
        fact_id = learn_client.facts.add("User prefers Python")
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]

        # Should not raise
        learn_client.facts.set_embedding(fact_id, embedding, model_name="test-model")

        # Verify by listing facts without embeddings
        without = learn_client.facts.list_without_embeddings("test-model")
        assert all(f.id != fact_id for f in without)

    def test_set_embedding_upsert(self, learn_client, clean_tables):
        """Test that set_embedding updates existing embedding."""
        fact_id = learn_client.facts.add("Test fact")
        embedding1 = [0.1, 0.2, 0.3]
        embedding2 = [0.4, 0.5, 0.6]

        learn_client.facts.set_embedding(fact_id, embedding1, model_name="test-model")
        learn_client.facts.set_embedding(fact_id, embedding2, model_name="test-model")

        # Should still only have one embedding for this model
        # Verify via search_similar - should find the fact with new embedding
        results = learn_client.facts.search_similar(
            embedding=[0.4, 0.5, 0.6],
            model_name="test-model",
            min_similarity=0.9,
        )
        assert len(results) == 1
        assert results[0].fact.id == fact_id

    def test_set_embedding_different_models(self, learn_client, clean_tables):
        """Test setting embeddings with different model names."""
        fact_id = learn_client.facts.add("Multi-model fact")
        embedding1 = [0.1, 0.2, 0.3]
        embedding2 = [0.4, 0.5, 0.6, 0.7]  # Different dimensions

        learn_client.facts.set_embedding(fact_id, embedding1, model_name="model-a")
        learn_client.facts.set_embedding(fact_id, embedding2, model_name="model-b")

        # Fact should not appear in list_without_embeddings for either model
        without_a = learn_client.facts.list_without_embeddings("model-a")
        without_b = learn_client.facts.list_without_embeddings("model-b")
        assert all(f.id != fact_id for f in without_a)
        assert all(f.id != fact_id for f in without_b)

        # But should appear for a different model
        without_c = learn_client.facts.list_without_embeddings("model-c")
        assert any(f.id == fact_id for f in without_c)

    def test_set_embedding_nonexistent_fact_raises(self, learn_client, clean_tables):
        """Test setting embedding for non-existent fact raises NotFoundError."""
        with pytest.raises(NotFoundError):
            learn_client.facts.set_embedding(99999, [0.1, 0.2], model_name="test")

    def test_set_embedding_empty_raises(self, learn_client, clean_tables):
        """Test setting empty embedding raises ValidationError."""
        from llm_learn import ValidationError

        fact_id = learn_client.facts.add("Test fact")
        with pytest.raises(ValidationError, match="cannot be empty"):
            learn_client.facts.set_embedding(fact_id, [], model_name="test")

    def test_set_embedding_invalid_values_raises(self, learn_client, clean_tables):
        """Test setting embedding with invalid values raises ValidationError."""
        import math

        from llm_learn import ValidationError

        fact_id = learn_client.facts.add("Test fact")

        with pytest.raises(ValidationError, match="must be finite"):
            learn_client.facts.set_embedding(fact_id, [0.1, math.nan, 0.3], model_name="test")

        with pytest.raises(ValidationError, match="must be finite"):
            learn_client.facts.set_embedding(fact_id, [0.1, math.inf, 0.3], model_name="test")

    def test_list_without_embeddings(self, learn_client, clean_tables):
        """Test listing facts without embeddings for a model."""
        fact1_id = learn_client.facts.add("Fact with embedding")
        fact2_id = learn_client.facts.add("Fact without embedding")
        fact3_id = learn_client.facts.add("Another without")

        learn_client.facts.set_embedding(fact1_id, [0.1, 0.2, 0.3], model_name="test-model")

        without = learn_client.facts.list_without_embeddings("test-model")

        fact_ids = [f.id for f in without]
        assert fact1_id not in fact_ids
        assert fact2_id in fact_ids
        assert fact3_id in fact_ids

    def test_list_without_embeddings_excludes_inactive(self, learn_client, clean_tables):
        """Test that list_without_embeddings excludes inactive facts."""
        fact1_id = learn_client.facts.add("Active fact")
        fact2_id = learn_client.facts.add("Inactive fact")
        learn_client.facts.deactivate(fact2_id)

        without = learn_client.facts.list_without_embeddings("test-model")

        fact_ids = [f.id for f in without]
        assert fact1_id in fact_ids
        assert fact2_id not in fact_ids

    def test_list_without_embeddings_respects_limit(self, learn_client, clean_tables):
        """Test list_without_embeddings respects limit parameter."""
        for i in range(10):
            learn_client.facts.add(f"Fact {i}")

        without = learn_client.facts.list_without_embeddings("test-model", limit=3)
        assert len(without) == 3

    def test_search_similar(self, learn_client, clean_tables):
        """Test searching facts by embedding similarity."""
        fact1_id = learn_client.facts.add("Python programming")
        fact2_id = learn_client.facts.add("JavaScript development")
        fact3_id = learn_client.facts.add("Database design")

        # Set embeddings - fact1 and fact2 similar, fact3 different
        learn_client.facts.set_embedding(fact1_id, [0.9, 0.1, 0.0], model_name="test")
        learn_client.facts.set_embedding(fact2_id, [0.8, 0.2, 0.0], model_name="test")
        learn_client.facts.set_embedding(fact3_id, [0.0, 0.0, 1.0], model_name="test")

        # Search for something similar to fact1
        results = learn_client.facts.search_similar(
            embedding=[0.85, 0.15, 0.0],
            model_name="test",
            top_k=10,
            min_similarity=0.5,
        )

        # Should find fact1 and fact2, not fact3
        assert len(results) >= 2
        result_ids = [r.fact.id for r in results]
        assert fact1_id in result_ids
        assert fact2_id in result_ids
        # fact3 might not be in results due to low similarity

        # Results should be sorted by similarity (highest first)
        similarities = [r.similarity for r in results]
        assert similarities == sorted(similarities, reverse=True)

    def test_search_similar_min_similarity_filter(self, learn_client, clean_tables):
        """Test that search_similar respects min_similarity threshold."""
        fact1_id = learn_client.facts.add("Very similar")
        fact2_id = learn_client.facts.add("Less similar")

        learn_client.facts.set_embedding(fact1_id, [1.0, 0.0, 0.0], model_name="test")
        learn_client.facts.set_embedding(fact2_id, [0.5, 0.5, 0.5], model_name="test")

        # High threshold - should only match very similar
        results = learn_client.facts.search_similar(
            embedding=[1.0, 0.0, 0.0],
            model_name="test",
            min_similarity=0.95,
        )

        result_ids = [r.fact.id for r in results]
        assert fact1_id in result_ids
        # fact2 should be filtered out due to low similarity

    def test_search_similar_top_k_limit(self, learn_client, clean_tables):
        """Test that search_similar respects top_k limit."""
        for i in range(10):
            fact_id = learn_client.facts.add(f"Fact {i}")
            learn_client.facts.set_embedding(fact_id, [0.5, 0.5, 0.0], model_name="test")

        results = learn_client.facts.search_similar(
            embedding=[0.5, 0.5, 0.0],
            model_name="test",
            top_k=3,
            min_similarity=0.0,
        )

        assert len(results) == 3

    def test_search_similar_active_only(self, learn_client, clean_tables):
        """Test that search_similar excludes inactive facts by default."""
        fact1_id = learn_client.facts.add("Active fact")
        fact2_id = learn_client.facts.add("Inactive fact")

        learn_client.facts.set_embedding(fact1_id, [1.0, 0.0], model_name="test")
        learn_client.facts.set_embedding(fact2_id, [1.0, 0.0], model_name="test")
        learn_client.facts.deactivate(fact2_id)

        # Default: active_only=True
        results = learn_client.facts.search_similar(
            embedding=[1.0, 0.0],
            model_name="test",
            min_similarity=0.0,
        )

        result_ids = [r.fact.id for r in results]
        assert fact1_id in result_ids
        assert fact2_id not in result_ids

        # With active_only=False
        results_all = learn_client.facts.search_similar(
            embedding=[1.0, 0.0],
            model_name="test",
            min_similarity=0.0,
            active_only=False,
        )

        result_ids_all = [r.fact.id for r in results_all]
        assert fact1_id in result_ids_all
        assert fact2_id in result_ids_all

    def test_search_similar_empty_results(self, learn_client, clean_tables):
        """Test search_similar returns empty list when no matches."""
        fact_id = learn_client.facts.add("Test fact")
        learn_client.facts.set_embedding(fact_id, [1.0, 0.0, 0.0], model_name="test")

        # Search with completely different embedding and high threshold
        results = learn_client.facts.search_similar(
            embedding=[0.0, 0.0, 1.0],
            model_name="test",
            min_similarity=0.99,
        )

        assert results == []

    def test_search_similar_wrong_model_returns_empty(self, learn_client, clean_tables):
        """Test search_similar returns empty when model doesn't match."""
        fact_id = learn_client.facts.add("Test fact")
        learn_client.facts.set_embedding(fact_id, [1.0, 0.0], model_name="model-a")

        results = learn_client.facts.search_similar(
            embedding=[1.0, 0.0],
            model_name="model-b",  # Different model
            min_similarity=0.0,
        )

        assert results == []

    def test_delete_fact_cascades_to_embeddings(self, learn_client, clean_tables, database):
        """Test that deleting a fact also deletes its embeddings."""
        fact_id = learn_client.facts.add("Fact to delete")
        learn_client.facts.set_embedding(fact_id, [0.1, 0.2], model_name="test")

        # Verify embedding exists
        results = learn_client.facts.search_similar(
            embedding=[0.1, 0.2],
            model_name="test",
            min_similarity=0.9,
        )
        assert len(results) == 1

        # Delete the fact
        learn_client.facts.delete(fact_id)

        # Embedding should be gone (cascade delete)
        results_after = learn_client.facts.search_similar(
            embedding=[0.1, 0.2],
            model_name="test",
            min_similarity=0.0,
        )
        assert len(results_after) == 0
