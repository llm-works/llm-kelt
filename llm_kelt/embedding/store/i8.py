# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-kelt Authors

"""Int8 embedding store using application-level scalar quantization."""

from __future__ import annotations

from ..quantize import quantize_int8
from ..types import Calibration, QuantizationFormat, QuantizedEmbedding
from .quantized import QuantizedStore


class Int8Store(QuantizedStore):
    """Int8 embedding store using application-level scalar quantization.

    Stores embeddings as quantized uint8 bytes with per-embedding scale/offset.
    Provides 4x storage reduction with ~99% accuracy retention.
    Search requires dequantization or approximate int8 distance computation.
    """

    _FORMAT = QuantizationFormat.I8

    def _quantize(
        self, embedding: list[float], calibration: Calibration | None
    ) -> QuantizedEmbedding:
        """Quantize to int8."""
        return quantize_int8(embedding, calibration)
