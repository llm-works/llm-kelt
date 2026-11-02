# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-kelt Authors

"""Int4 embedding store using application-level scalar quantization."""

from __future__ import annotations

from ..quantize import quantize_int4
from ..types import Calibration, QuantizationFormat, QuantizedEmbedding
from .quantized import QuantizedStore


class Int4Store(QuantizedStore):
    """Int4 embedding store using application-level scalar quantization.

    Stores embeddings as packed 4-bit values (2 values per byte) with per-embedding scale/offset.
    Provides 8x storage reduction with ~95-97% accuracy retention.
    Recommended to use with reranking for better accuracy.
    """

    _FORMAT = QuantizationFormat.I4

    def _quantize(
        self, embedding: list[float], calibration: Calibration | None
    ) -> QuantizedEmbedding:
        """Quantize to packed int4."""
        return quantize_int4(embedding, calibration)
