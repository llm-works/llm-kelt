"""Unit tests for embedding validation."""

import pytest

from llm_kelt.core.embedding import _validate_embedding
from llm_kelt.core.exceptions import ValidationError


class TestValidateEmbedding:
    """Tests for _validate_embedding function."""

    def test_valid_embedding(self):
        """Valid embedding should not raise."""
        _validate_embedding([0.1, 0.2, 0.3])

    def test_valid_embedding_with_integers(self):
        """Integers should be accepted."""
        _validate_embedding([1, 2, 3])

    def test_valid_embedding_mixed_types(self):
        """Mixed int and float should be accepted."""
        _validate_embedding([1, 0.5, 2])

    def test_empty_embedding_raises(self):
        """Empty list should raise ValidationError."""
        with pytest.raises(ValidationError, match="cannot be empty"):
            _validate_embedding([])

    def test_none_embedding_raises(self):
        """None should raise ValidationError."""
        with pytest.raises(ValidationError, match="cannot be empty"):
            _validate_embedding(None)  # type: ignore

    def test_non_numeric_raises(self):
        """Non-numeric values should raise ValidationError."""
        with pytest.raises(ValidationError, match="must be numeric"):
            _validate_embedding([0.1, "not a number", 0.3])

    def test_nan_raises(self):
        """NaN values should raise ValidationError."""
        with pytest.raises(ValidationError, match="must be finite"):
            _validate_embedding([0.1, float("nan"), 0.3])

    def test_positive_infinity_raises(self):
        """Positive infinity should raise ValidationError."""
        with pytest.raises(ValidationError, match="must be finite"):
            _validate_embedding([0.1, float("inf"), 0.3])

    def test_negative_infinity_raises(self):
        """Negative infinity should raise ValidationError."""
        with pytest.raises(ValidationError, match="must be finite"):
            _validate_embedding([0.1, float("-inf"), 0.3])
