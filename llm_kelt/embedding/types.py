"""Core types for embedding quantization."""

import re
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar


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

    Args:
        context_key: Context key for isolation (used by adapters, not table naming).
        format: Quantization format (F32, F16, I8, I4).
        dimensions: Vector dimensions.
        prefix: Optional table prefix for custom tables. Results in
            embeddings_{prefix}_{dimensions}_{format}. If None, uses
            embeddings_{dimensions}_{format}.
    """

    context_key: str
    format: QuantizationFormat = QuantizationFormat.F16
    dimensions: int = 384
    prefix: str | None = None

    _PREFIX_PATTERN: ClassVar[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9_]+$")

    def __post_init__(self) -> None:
        if self.dimensions <= 0:
            raise ValueError(f"dimensions must be positive, got {self.dimensions}")
        if self.prefix is not None and not self._PREFIX_PATTERN.match(self.prefix):
            raise ValueError(f"prefix must be alphanumeric/underscore, got {self.prefix!r}")

    @property
    def table_name(self) -> str:
        """Get table name for this config."""
        if self.prefix:
            return f"embeddings_{self.prefix}_{self.dimensions}_{self.format.value}"
        return f"embeddings_{self.dimensions}_{self.format.value}"
