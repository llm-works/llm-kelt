# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-kelt Authors

"""Quantization and dequantization algorithms for embeddings."""

from __future__ import annotations

import struct
from typing import TYPE_CHECKING

from .types import Calibration, QuantizationFormat, QuantizedEmbedding

if TYPE_CHECKING:
    pass


def quantize_int8(
    embedding: list[float],
    calibration: Calibration | None = None,
) -> QuantizedEmbedding:
    """Quantize float32 embedding to int8 using scalar quantization.

    Args:
        embedding: Float32 embedding vector.
        calibration: Optional calibration data. If None, uses per-embedding min/max.

    Returns:
        QuantizedEmbedding with int8 data (1 byte per dimension).
    """
    if calibration:
        cal_min, cal_max = calibration.min_val, calibration.max_val
    else:
        cal_min, cal_max = min(embedding), max(embedding)

    val_range = cal_max - cal_min
    if val_range == 0:
        scale = 0.0
        offset = cal_min
        quantized = bytes([0] * len(embedding))
    else:
        scale = val_range / 255.0
        offset = cal_min
        quantized_values = []
        for val in embedding:
            clamped = max(cal_min, min(cal_max, val))
            q = int(round((clamped - offset) / scale))
            q = max(0, min(255, q))
            quantized_values.append(q)
        quantized = bytes(quantized_values)

    return QuantizedEmbedding(
        data=quantized,
        format=QuantizationFormat.I8,
        dimensions=len(embedding),
        scale=scale,
        offset=offset,
    )


def dequantize_int8(qemb: QuantizedEmbedding) -> list[float]:
    """Dequantize int8 embedding back to float32.

    Args:
        qemb: Quantized embedding with int8 data.

    Returns:
        Float32 embedding vector.
    """
    if qemb.format != QuantizationFormat.I8:
        raise ValueError(f"Expected I8 format, got {qemb.format}")
    return [b * qemb.scale + qemb.offset for b in qemb.data]


def _pack_nibbles(nibbles: list[int]) -> bytes:
    """Pack 4-bit nibbles into bytes (2 per byte, high nibble first)."""
    packed = []
    for i in range(0, len(nibbles), 2):
        high = nibbles[i] << 4
        low = nibbles[i + 1] if i + 1 < len(nibbles) else 0
        packed.append(high | low)
    return bytes(packed)


def quantize_int4(
    embedding: list[float],
    calibration: Calibration | None = None,
) -> QuantizedEmbedding:
    """Quantize float32 embedding to int4 (4 bits per value, packed)."""
    cal_min, cal_max = (
        (calibration.min_val, calibration.max_val)
        if calibration
        else (min(embedding), max(embedding))
    )
    val_range = cal_max - cal_min

    if val_range == 0:
        nibbles = [0] * len(embedding)
        return QuantizedEmbedding(
            data=_pack_nibbles(nibbles),
            format=QuantizationFormat.I4,
            dimensions=len(embedding),
            scale=0.0,
            offset=cal_min,
        )

    scale = val_range / 15.0
    nibbles = [
        max(0, min(15, int(round((max(cal_min, min(cal_max, v)) - cal_min) / scale))))
        for v in embedding
    ]
    return QuantizedEmbedding(
        data=_pack_nibbles(nibbles),
        format=QuantizationFormat.I4,
        dimensions=len(embedding),
        scale=scale,
        offset=cal_min,
    )


def dequantize_int4(qemb: QuantizedEmbedding) -> list[float]:
    """Dequantize int4 embedding back to float32.

    Args:
        qemb: Quantized embedding with packed int4 data.

    Returns:
        Float32 embedding vector.
    """
    if qemb.format != QuantizationFormat.I4:
        raise ValueError(f"Expected I4 format, got {qemb.format}")

    values = []
    for byte in qemb.data:
        high = (byte >> 4) & 0x0F
        low = byte & 0x0F
        values.append(high * qemb.scale + qemb.offset)
        values.append(low * qemb.scale + qemb.offset)

    return values[: qemb.dimensions]


def quantize(
    embedding: list[float],
    fmt: QuantizationFormat,
    calibration: Calibration | None = None,
) -> QuantizedEmbedding | list[float]:
    """Quantize embedding to specified format.

    Args:
        embedding: Float32 embedding vector.
        fmt: Target quantization format.
        calibration: Optional calibration data (required for I8/I4 with model-level calibration).

    Returns:
        QuantizedEmbedding for I8/I4, or original list for F32/F16 (native pgvector).
    """
    if fmt == QuantizationFormat.F32:
        return embedding
    elif fmt == QuantizationFormat.F16:
        return embedding  # pgvector handles conversion
    elif fmt == QuantizationFormat.I8:
        return quantize_int8(embedding, calibration)
    elif fmt == QuantizationFormat.I4:
        return quantize_int4(embedding, calibration)
    else:
        raise ValueError(f"Unknown format: {fmt}")


def dequantize(qemb: QuantizedEmbedding) -> list[float]:
    """Dequantize embedding back to float32.

    Args:
        qemb: Quantized embedding.

    Returns:
        Float32 embedding vector.
    """
    if qemb.format == QuantizationFormat.I8:
        return dequantize_int8(qemb)
    elif qemb.format == QuantizationFormat.I4:
        return dequantize_int4(qemb)
    else:
        raise ValueError(f"Cannot dequantize format: {qemb.format}")


def pack_scale_offset(scale: float, offset: float) -> bytes:
    """Pack scale and offset into bytes for storage alongside quantized data."""
    return struct.pack("<ff", scale, offset)


def unpack_scale_offset(data: bytes) -> tuple[float, float]:
    """Unpack scale and offset from bytes."""
    return struct.unpack("<ff", data)
