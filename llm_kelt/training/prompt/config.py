# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-kelt Authors

"""Prompt tuning configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class Config:
    """Prompt tuning configuration.

    Prompt tuning adds a small set of learnable "virtual tokens" to the input.
    The entire base model is frozen - only these embeddings are trained.
    This is extremely parameter-efficient (~50K params vs ~50M for LoRA r=32).

    Attributes:
        num_virtual_tokens: Number of learnable tokens prepended to input.
            More tokens = more capacity but slower inference. 20 is a good default.
        prompt_tuning_init: Initialization strategy.
            "TEXT" initializes from a text string (recommended).
            "RANDOM" uses random initialization.
        prompt_tuning_init_text: Text to initialize from when using TEXT init.
            Should describe the desired behavior/style.
        task_type: Task type for PEFT. "CAUSAL_LM" for language modeling.
    """

    num_virtual_tokens: int = 20
    prompt_tuning_init: Literal["TEXT", "RANDOM"] = "TEXT"
    prompt_tuning_init_text: str = "You are a helpful assistant."
    task_type: Literal["CAUSAL_LM", "SEQ_CLS", "SEQ_2_SEQ_LM", "TOKEN_CLS"] = "CAUSAL_LM"

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.num_virtual_tokens <= 0:
            raise ValueError(f"num_virtual_tokens must be positive, got {self.num_virtual_tokens}")
        if self.prompt_tuning_init not in ("TEXT", "RANDOM"):
            raise ValueError(
                f"prompt_tuning_init must be 'TEXT' or 'RANDOM', got {self.prompt_tuning_init}"
            )
        if self.prompt_tuning_init == "TEXT" and not self.prompt_tuning_init_text:
            raise ValueError("prompt_tuning_init_text is required when prompt_tuning_init='TEXT'")
        if self.task_type not in ("CAUSAL_LM", "SEQ_CLS", "SEQ_2_SEQ_LM", "TOKEN_CLS"):
            raise ValueError(f"Invalid task_type: {self.task_type}")
