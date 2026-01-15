"""Unit tests for facts module utility functions."""

import math
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from llm_learn.collection.facts import (
    ScoredFact,
    _get_similarity_query,
    _row_to_fact,
    _validate_embedding,
)
from llm_learn.core.exceptions import ValidationError
from llm_learn.core.models import Fact


class TestScoredFact:
    """Test ScoredFact dataclass."""

    def test_create_scored_fact(self):
        """Test creating a ScoredFact."""
        fact = Fact(
            id=1,
            profile_id=1,
            content="Test content",
            category="preferences",
            source="user",
            confidence=0.9,
            active=True,
        )
        scored = ScoredFact(fact=fact, similarity=0.85)

        assert scored.fact == fact
        assert scored.similarity == 0.85

    def test_scored_fact_immutable_attributes(self):
        """Test ScoredFact attributes are accessible."""
        fact = Fact(id=1, profile_id=1, content="Test", source="user")
        scored = ScoredFact(fact=fact, similarity=0.5)

        assert scored.fact.content == "Test"
        assert scored.similarity == 0.5


class TestRowToFact:
    """Test _row_to_fact conversion function."""

    def test_convert_row_to_fact(self):
        """Test converting a database row to Fact object."""
        now = datetime.now(UTC)
        row = SimpleNamespace(
            id=42,
            profile_id=1,
            content="User prefers Python",
            category="preferences",
            source="user",
            confidence=0.95,
            active=True,
            created_at=now,
            updated_at=None,
        )

        fact = _row_to_fact(row)

        assert isinstance(fact, Fact)
        assert fact.id == 42
        assert fact.profile_id == 1
        assert fact.content == "User prefers Python"
        assert fact.category == "preferences"
        assert fact.source == "user"
        assert fact.confidence == 0.95
        assert fact.active is True
        assert fact.created_at == now
        assert fact.updated_at is None

    def test_convert_row_with_updated_at(self):
        """Test converting row with updated_at timestamp."""
        created = datetime(2024, 1, 1, tzinfo=UTC)
        updated = datetime(2024, 1, 15, tzinfo=UTC)
        row = SimpleNamespace(
            id=1,
            profile_id=1,
            content="Updated fact",
            category=None,
            source="inferred",
            confidence=0.7,
            active=True,
            created_at=created,
            updated_at=updated,
        )

        fact = _row_to_fact(row)

        assert fact.created_at == created
        assert fact.updated_at == updated

    def test_convert_row_with_null_category(self):
        """Test converting row with null category."""
        row = SimpleNamespace(
            id=1,
            profile_id=1,
            content="Uncategorized fact",
            category=None,
            source="user",
            confidence=1.0,
            active=True,
            created_at=datetime.now(UTC),
            updated_at=None,
        )

        fact = _row_to_fact(row)

        assert fact.category is None


class TestValidateEmbedding:
    """Test _validate_embedding validation function."""

    def test_valid_embedding(self):
        """Test valid embedding passes validation."""
        _validate_embedding([0.1, 0.2, 0.3])  # Should not raise

    def test_valid_embedding_with_integers(self):
        """Test embedding with integers passes validation."""
        _validate_embedding([1, 2, 3])  # Should not raise

    def test_valid_embedding_with_mixed_types(self):
        """Test embedding with mixed int/float passes validation."""
        _validate_embedding([1, 0.5, 2, 0.25])  # Should not raise

    def test_empty_embedding_raises(self):
        """Test empty embedding raises ValidationError."""
        with pytest.raises(ValidationError, match="cannot be empty"):
            _validate_embedding([])

    def test_nan_raises(self):
        """Test NaN value raises ValidationError."""
        with pytest.raises(ValidationError, match="must be finite"):
            _validate_embedding([0.1, math.nan, 0.3])

    def test_inf_raises(self):
        """Test infinity value raises ValidationError."""
        with pytest.raises(ValidationError, match="must be finite"):
            _validate_embedding([0.1, math.inf, 0.3])

    def test_negative_inf_raises(self):
        """Test negative infinity raises ValidationError."""
        with pytest.raises(ValidationError, match="must be finite"):
            _validate_embedding([0.1, -math.inf, 0.3])

    def test_non_numeric_raises(self):
        """Test non-numeric value raises ValidationError."""
        with pytest.raises(ValidationError, match="must be numeric"):
            _validate_embedding([0.1, "string", 0.3])  # type: ignore[list-item]

    def test_none_value_raises(self):
        """Test None value raises ValidationError."""
        with pytest.raises(ValidationError, match="must be numeric"):
            _validate_embedding([0.1, None, 0.3])  # type: ignore[list-item]


class TestGetSimilarityQuery:
    """Test _get_similarity_query SQL selector."""

    def test_query_with_active_only_true(self):
        """Test query includes active filter when active_only=True."""
        query = _get_similarity_query(active_only=True)
        query_text = str(query)

        assert "f.active = true" in query_text
        assert "FROM facts f" in query_text
        assert "JOIN fact_embeddings e ON e.fact_id = f.id" in query_text
        assert "ORDER BY e.embedding <=>" in query_text
        assert "LIMIT :top_k" in query_text

    def test_query_with_active_only_false(self):
        """Test query excludes active filter when active_only=False."""
        query = _get_similarity_query(active_only=False)
        query_text = str(query)

        assert "f.active = true" not in query_text
        # Should still have other filters
        assert "f.profile_id = :profile_id" in query_text
        assert "e.model_name = :model_name" in query_text

    def test_query_has_required_parameters(self):
        """Test query expects required parameters."""
        query = _get_similarity_query(active_only=True)
        query_text = str(query)

        # Check for parameter placeholders
        assert ":embedding" in query_text
        assert ":profile_id" in query_text
        assert ":model_name" in query_text
        assert ":min_similarity" in query_text
        assert ":top_k" in query_text

    def test_query_selects_similarity_score(self):
        """Test query calculates similarity score."""
        query = _get_similarity_query(active_only=True)
        query_text = str(query)

        # Should calculate 1 - cosine_distance as similarity
        assert "1 - (e.embedding <=>" in query_text
        assert "as similarity" in query_text

    def test_query_filters_by_min_similarity(self):
        """Test query filters by minimum similarity threshold."""
        query = _get_similarity_query(active_only=True)
        query_text = str(query)

        assert ">= :min_similarity" in query_text
