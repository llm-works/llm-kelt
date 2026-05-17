"""Unit tests for embedding quantization."""

import pytest

from llm_kelt.embedding import (
    Calibration,
    EmbeddingConfig,
    QuantizationFormat,
    dequantize,
    dequantize_int4,
    dequantize_int8,
    quantize_int4,
    quantize_int8,
)


class TestQuantizeInt8:
    """Tests for int8 scalar quantization."""

    def test_quantize_basic(self) -> None:
        """Quantize a simple embedding."""
        embedding = [0.0, 0.5, 1.0]
        result = quantize_int8(embedding)

        assert result.format == QuantizationFormat.I8
        assert result.dimensions == 3
        assert len(result.data) == 3

    def test_roundtrip(self) -> None:
        """Quantize and dequantize should approximately recover original values."""
        embedding = [0.1, 0.3, 0.5, 0.7, 0.9]
        quantized = quantize_int8(embedding)
        recovered = dequantize_int8(quantized)

        assert len(recovered) == len(embedding)
        for orig, rec in zip(embedding, recovered):
            assert abs(orig - rec) < 0.01  # Within 1% error

    def test_negative_values(self) -> None:
        """Quantize embeddings with negative values."""
        embedding = [-1.0, -0.5, 0.0, 0.5, 1.0]
        quantized = quantize_int8(embedding)
        recovered = dequantize_int8(quantized)

        for orig, rec in zip(embedding, recovered):
            assert abs(orig - rec) < 0.01

    def test_uniform_embedding(self) -> None:
        """Quantize embedding where all values are the same."""
        embedding = [0.5, 0.5, 0.5, 0.5]
        quantized = quantize_int8(embedding)
        recovered = dequantize_int8(quantized)

        for rec in recovered:
            assert abs(rec - 0.5) < 0.01

    def test_with_calibration(self) -> None:
        """Quantize with external calibration range."""
        embedding = [0.2, 0.4, 0.6]
        calibration = Calibration(min_val=0.0, max_val=1.0, model_name="test", dimensions=3)
        quantized = quantize_int8(embedding, calibration)
        recovered = dequantize_int8(quantized)

        for orig, rec in zip(embedding, recovered):
            assert abs(orig - rec) < 0.01

    def test_storage_size(self) -> None:
        """Int8 should use 1 byte per dimension."""
        embedding = [0.1] * 384
        quantized = quantize_int8(embedding)
        assert quantized.storage_bytes() == 384


class TestQuantizeInt4:
    """Tests for int4 scalar quantization."""

    def test_quantize_basic(self) -> None:
        """Quantize a simple embedding to int4."""
        embedding = [0.0, 0.5, 1.0, 0.25]
        result = quantize_int4(embedding)

        assert result.format == QuantizationFormat.I4
        assert result.dimensions == 4
        assert len(result.data) == 2  # Packed: 2 values per byte

    def test_roundtrip(self) -> None:
        """Int4 roundtrip with slightly higher tolerance."""
        embedding = [0.0, 0.25, 0.5, 0.75, 1.0]
        quantized = quantize_int4(embedding)
        recovered = dequantize_int4(quantized)

        assert len(recovered) == len(embedding)
        for orig, rec in zip(embedding, recovered):
            assert abs(orig - rec) < 0.1  # Within 10% error (only 16 levels)

    def test_odd_length(self) -> None:
        """Quantize odd-length embedding (padded)."""
        embedding = [0.1, 0.2, 0.3]
        quantized = quantize_int4(embedding)
        recovered = dequantize_int4(quantized)

        assert len(recovered) == 3

    def test_storage_size(self) -> None:
        """Int4 should use 0.5 bytes per dimension (packed)."""
        embedding = [0.1] * 384
        quantized = quantize_int4(embedding)
        assert quantized.storage_bytes() == 192  # 384 / 2

    def test_storage_size_odd(self) -> None:
        """Int4 with odd dimensions rounds up."""
        embedding = [0.1] * 385
        quantized = quantize_int4(embedding)
        assert quantized.storage_bytes() == 193  # ceil(385 / 2)


class TestDequantize:
    """Tests for generic dequantize function."""

    def test_dequantize_int8(self) -> None:
        """Generic dequantize dispatches to int8."""
        embedding = [0.2, 0.4, 0.6]
        quantized = quantize_int8(embedding)
        recovered = dequantize(quantized)
        assert len(recovered) == 3

    def test_dequantize_int4(self) -> None:
        """Generic dequantize dispatches to int4."""
        embedding = [0.2, 0.4, 0.6, 0.8]
        quantized = quantize_int4(embedding)
        recovered = dequantize(quantized)
        assert len(recovered) == 4


class TestQuantizationFormat:
    """Tests for QuantizationFormat enum."""

    def test_bytes_per_dim(self) -> None:
        """Check storage bytes per dimension."""
        assert QuantizationFormat.F32.bytes_per_dim == 4.0
        assert QuantizationFormat.F16.bytes_per_dim == 2.0
        assert QuantizationFormat.I8.bytes_per_dim == 1.0
        assert QuantizationFormat.I4.bytes_per_dim == 0.5

    def test_is_native_pgvector(self) -> None:
        """Check which formats are native pgvector."""
        assert QuantizationFormat.F32.is_native_pgvector is True
        assert QuantizationFormat.F16.is_native_pgvector is True
        assert QuantizationFormat.I8.is_native_pgvector is False
        assert QuantizationFormat.I4.is_native_pgvector is False

    def test_table_name(self) -> None:
        """Check table name generation."""
        assert QuantizationFormat.F32.table_name(384) == "embeddings_384_f32"
        assert QuantizationFormat.F16.table_name(1536) == "embeddings_1536_f16"
        assert QuantizationFormat.I8.table_name(384) == "embeddings_384_i8"
        assert QuantizationFormat.I4.table_name(768) == "embeddings_768_i4"


class TestEmbeddingConfig:
    """Tests for EmbeddingConfig dataclass."""

    def test_default_config(self) -> None:
        """Default config uses f32."""
        config = EmbeddingConfig(context_key="test")
        assert config.primary_format == QuantizationFormat.F32
        assert config.search_format == QuantizationFormat.F32
        assert config.store_formats == [QuantizationFormat.F32]
        assert config.rerank_format is None

    def test_table_names(self) -> None:
        """Generate table names for all configured formats."""
        config = EmbeddingConfig(
            context_key="test",
            store_formats=[QuantizationFormat.F32, QuantizationFormat.I8],
            search_format=QuantizationFormat.I8,
            dimensions=384,
        )
        names = config.table_names()
        assert names == ["embeddings_384_f32", "embeddings_384_i8"]

    def test_validation_search_in_store(self) -> None:
        """Search format must be in store formats."""
        with pytest.raises(ValueError, match="search_format.*must be in store_formats"):
            EmbeddingConfig(
                context_key="test",
                store_formats=[QuantizationFormat.F32],
                search_format=QuantizationFormat.I8,
            )

    def test_validation_rerank_in_store(self) -> None:
        """Rerank format must be in store formats."""
        with pytest.raises(ValueError, match="rerank_format.*must be in store_formats"):
            EmbeddingConfig(
                context_key="test",
                store_formats=[QuantizationFormat.I8],
                search_format=QuantizationFormat.I8,
                rerank_format=QuantizationFormat.F32,
            )

    def test_validation_rerank_oversample(self) -> None:
        """Rerank oversample must be >= 2 when rerank is set."""
        with pytest.raises(ValueError, match="rerank_oversample must be >= 2"):
            EmbeddingConfig(
                context_key="test",
                store_formats=[QuantizationFormat.F32, QuantizationFormat.I8],
                search_format=QuantizationFormat.I8,
                rerank_format=QuantizationFormat.F32,
                rerank_oversample=1,
            )
