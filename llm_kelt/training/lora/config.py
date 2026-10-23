# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-kelt Authors

"""LoRA adapter configuration."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

# Default target modules for Qwen2.5 architecture
QWEN_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

# Qwen3.5 / Qwen3.8 (model_type qwen3_5) mix two kinds of layers: full-attention layers use
# q/k/v/o_proj, Gated DeltaNet (linear attention) layers use in_proj_qkv, in_proj_z and
# out_proj. in_proj_a/in_proj_b are tiny per-head projections and are left out.
QWEN3_5_TARGET_MODULES = [*QWEN_TARGET_MODULES, "in_proj_qkv", "in_proj_z", "out_proj"]

# The vision tower and the multi-token-prediction head reuse the language model's projection
# names; LoRA adapts the language model only. Regex over full module names.
QWEN3_5_EXCLUDE_MODULES = r"(.*\.)?(visual|mtp)\..*"

# model_type -> (target_modules, exclude_modules). Other model types use QWEN_TARGET_MODULES
# with no exclusion.
_FAMILY_DEFAULTS: dict[str, tuple[list[str], str | None]] = {
    "qwen3_5": (QWEN3_5_TARGET_MODULES, QWEN3_5_EXCLUDE_MODULES),
    "qwen3_5_text": (QWEN3_5_TARGET_MODULES, QWEN3_5_EXCLUDE_MODULES),
}


def default_lora_modules(model_type: str) -> tuple[list[str], str | None]:
    """Return the default (target_modules, exclude_modules) for a transformers model_type."""
    targets, exclude = _FAMILY_DEFAULTS.get(model_type, (QWEN_TARGET_MODULES, None))
    return list(targets), exclude


@dataclass
class Config:
    """LoRA adapter configuration.

    Attributes:
        r: LoRA rank. Higher = more capacity but more VRAM. 16 is a good default.
        lora_alpha: Scaling factor. Usually 2x rank.
        lora_dropout: Dropout for LoRA layers. 0.05 prevents overfitting.
        target_modules: Which modules to apply LoRA to. None = the model family's default
            (see default_lora_modules), filled in by resolve() once the model is loaded.
        exclude_modules: Regex of module names LoRA must not touch. None = the model
            family's default.
        bias: Bias training mode. "none" is most memory efficient.
        task_type: Task type for PEFT. "CAUSAL_LM" for language modeling.
        use_rslora: Use rank-stabilized scaling (alpha/sqrt(r) instead of alpha/r).
            Prevents gradient collapse on large models (32B+). See arXiv:2312.03732.
    """

    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] | None = None
    exclude_modules: str | None = None
    bias: Literal["none", "all", "lora_only"] = "none"
    task_type: Literal["CAUSAL_LM", "SEQ_CLS", "SEQ_2_SEQ_LM", "TOKEN_CLS"] = "CAUSAL_LM"
    use_rslora: bool = False

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.r <= 0:
            raise ValueError(f"LoRA rank must be positive, got {self.r}")
        if self.lora_alpha <= 0:
            raise ValueError(f"LoRA alpha must be positive, got {self.lora_alpha}")
        if not 0.0 <= self.lora_dropout < 1.0:
            raise ValueError(f"LoRA dropout must be in [0, 1), got {self.lora_dropout}")
        if self.target_modules is not None and not self.target_modules:
            raise ValueError("target_modules cannot be empty")
        if self.bias not in ("none", "all", "lora_only"):
            raise ValueError(f"bias must be 'none', 'all', or 'lora_only', got {self.bias}")
        if self.task_type not in ("CAUSAL_LM", "SEQ_CLS", "SEQ_2_SEQ_LM", "TOKEN_CLS"):
            raise ValueError(f"Invalid task_type: {self.task_type}")

    def resolve(self, model_type: str) -> Config:
        """Return a copy with unset target/exclude modules filled in for the model family.

        Args:
            model_type: The loaded model's ``config.model_type`` (e.g. "qwen2", "qwen3_5").
        """
        targets, exclude = default_lora_modules(model_type)
        return replace(
            self,
            target_modules=list(self.target_modules) if self.target_modules else targets,
            exclude_modules=exclude if self.exclude_modules is None else self.exclude_modules,
        )
