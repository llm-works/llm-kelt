"""Core types for embedding quantization."""

from dataclasses import dataclass
from enum import Enum


class QuantizationFormat(Enum):
    """Supported embedding quantization formats."""

    F32 = "f32"  # float32 - pgvector vector (full precision)
    F16 = "f16"  # float16 - pgvector halfvec (2x compression)
    I8 = "i8"  # int8 - application quantized (4x compression)
    I4 = "i4"  # int4 - application quantized (8x compression)

    @property
    def bytes_per_dim(self) -> float:
        """Storage bytes per dimension."""
        return {
            QuantizationFormat.F32: 4.0,
            QuantizationFormat.F16: 2.0,
            QuantizationFormat.I8: 1.0,
            QuantizationFormat.I4: 0.5,
        }[self]

    @property
    def is_native_pgvector(self) -> bool:
        """Whether this format uses native pgvector types."""
        return self in (QuantizationFormat.F32, QuantizationFormat.F16)

    def table_name(self, dimensions: int) -> str:
        """Generate table name for this format and dimension."""
        return f"embeddings_{dimensions}_{self.value}"


@dataclass
class QuantizedEmbedding:
    """Quantized embedding with metadata for dequantization."""

    data: bytes
    format: QuantizationFormat
    dimensions: int
    scale: float = 1.0
    offset: float = 0.0

    def storage_bytes(self) -> int:
        """Total storage size in bytes."""
        return len(self.data)


@dataclass
class Calibration:
    """Calibration data for quantization."""

    min_val: float
    max_val: float
    model_name: str
    dimensions: int
    sample_count: int = 0

    @property
    def range(self) -> float:
        """Value range."""
        return self.max_val - self.min_val


@dataclass
class Config:
    """Embedding configuration.

    Each config defines ONE format and dimension. Want multiple formats?
    Create multiple clients with different configs.
    """

    context_key: str
    format: QuantizationFormat = QuantizationFormat.F16
    dimensions: int = 384

    @property
    def table_name(self) -> str:
        """Get table name for this config."""
        return self.format.table_name(self.dimensions)
