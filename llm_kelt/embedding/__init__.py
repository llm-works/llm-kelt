"""Embedding quantization framework - multi-format vector storage."""

from .quantize import (
    dequantize,
    dequantize_int4,
    dequantize_int8,
    quantize,
    quantize_int4,
    quantize_int8,
)
from .router import EmbeddingRouter
from .types import Calibration, EmbeddingConfig, QuantizationFormat, QuantizedEmbedding

__all__ = [
    # Types
    "Calibration",
    "EmbeddingConfig",
    "QuantizationFormat",
    "QuantizedEmbedding",
    # Router
    "EmbeddingRouter",
    # Quantization functions
    "dequantize",
    "dequantize_int4",
    "dequantize_int8",
    "quantize",
    "quantize_int4",
    "quantize_int8",
]
