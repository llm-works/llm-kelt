# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-kelt Authors

"""Tiny random Qwen3.5 checkpoints (bf16 and compressed-tensors W4A16) for e2e training tests.

The model keeps the architecture's shape: 3 linear-attention (Gated DeltaNet) layers, 1
full-attention layer, a vision tower, saved as Qwen3_5ForConditionalGeneration. Layer widths
are multiples of 128 so int4 group quantization (group size 128) applies to every quantized
layer.

The compressed-tensors copy mirrors the published Qwen3.5 int4 checkpoints: linear-attention
projections, the vision tower and lm_head stay bf16 through an ignore list written with
``model.language_model.*`` names; all other linear layers are packed int4.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

TEXT_DIMS: dict[str, Any] = dict(
    hidden_size=128,
    intermediate_size=256,
    num_hidden_layers=4,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=32,
    linear_key_head_dim=16,
    linear_value_head_dim=16,
    linear_num_key_heads=2,
    linear_num_value_heads=4,
    linear_conv_kernel_dim=4,
    layer_types=["linear_attention"] * 3 + ["full_attention"],
    full_attention_interval=4,
    tie_word_embeddings=False,
)
VISION_DIMS: dict[str, Any] = dict(
    depth=1,
    hidden_size=32,
    intermediate_size=64,
    num_heads=2,
    out_hidden_size=128,
    num_position_embeddings=64,
)


def _tiny_config(vocab: int, eos_token_id: int) -> Any:
    """Qwen3_5Config with the tiny dimensions; the last 4 token ids are the vision tokens."""
    from transformers import Qwen3_5Config

    return Qwen3_5Config(
        text_config={**TEXT_DIMS, "vocab_size": vocab, "eos_token_id": eos_token_id},
        vision_config=VISION_DIMS,
        image_token_id=vocab - 4,
        video_token_id=vocab - 3,
        vision_start_token_id=vocab - 2,
        vision_end_token_id=vocab - 1,
        tie_word_embeddings=False,
    )


def build_bf16(tokenizer_src: Path, dst: Path) -> Path:
    """Save a tiny random Qwen3_5ForConditionalGeneration (bf16) with tokenizer_src's tokenizer."""
    import torch
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_src)
    config = _tiny_config(len(tokenizer), tokenizer.eos_token_id)
    torch.manual_seed(0)
    model = Qwen3_5ForConditionalGeneration(config).to(torch.bfloat16)
    model.save_pretrained(dst)
    tokenizer.save_pretrained(dst)
    return dst


def _calibrate_weight_scales(model) -> None:
    """Set symmetric int4 group scales from the weights (weight-only, no calibration data)."""
    for module in model.modules():
        scheme = getattr(module, "quantization_scheme", None)
        if scheme is None or not hasattr(module, "weight_scale"):
            continue
        weight = module.weight.data.float()
        groups = weight.reshape(weight.shape[0], -1, scheme.weights.group_size)
        scale = groups.abs().amax(dim=-1).clamp(min=1e-8) / 7.0
        module.weight_scale.data.copy_(scale.to(module.weight_scale.dtype))
        if hasattr(module, "weight_zero_point"):
            module.weight_zero_point.data.zero_()


def build_compressed(bf16_src: Path, dst: Path) -> Path:
    """Save a compressed-tensors W4A16 (pack-quantized) copy of a tiny bf16 Qwen3.5."""
    import torch
    from compressed_tensors.compressors import ModelCompressor
    from compressed_tensors.quantization import (
        QuantizationConfig,
        QuantizationStatus,
        apply_quantization_config,
        preset_name_to_scheme,
    )
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

    model = Qwen3_5ForConditionalGeneration.from_pretrained(bf16_src, dtype=torch.bfloat16)
    config = QuantizationConfig(
        config_groups={"group_0": preset_name_to_scheme("W4A16", ["Linear"])},
        ignore=["lm_head", "re:.*visual.*", "re:.*linear_attn.*"],
        quantization_status=QuantizationStatus.FROZEN,
    )
    apply_quantization_config(model, config)
    _calibrate_weight_scales(model)

    compressor = ModelCompressor.from_pretrained_model(model, quantization_format="pack-quantized")
    compressor.compress_model(model)
    model.save_pretrained(dst)
    compressor.update_config(dst)
    AutoTokenizer.from_pretrained(bf16_src).save_pretrained(dst)
    return dst
