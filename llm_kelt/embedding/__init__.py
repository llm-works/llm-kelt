# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-kelt Authors

"""Embedding quantization framework - multi-format vector storage."""

from .client import StoreClient
from .factory import Factory, get_factory
from .quantize import (
    dequantize,
    dequantize_int4,
    dequantize_int8,
    quantize,
    quantize_int4,
    quantize_int8,
)
from .types import Calibration, Config, QuantizationFormat, QuantizedEmbedding

__all__ = [
    # Client and Factory
    "StoreClient",
    "Factory",
    "get_factory",
    # Types
    "Calibration",
    "Config",
    "QuantizationFormat",
    "QuantizedEmbedding",
    # Quantization functions
    "dequantize",
    "dequantize_int4",
    "dequantize_int8",
    "quantize",
    "quantize_int4",
    "quantize_int8",
]
