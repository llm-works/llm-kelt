# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-kelt Authors

"""Prompt tuning using supervised fine-tuning (SFT).

Uses TRL's SFTTrainer with PEFT PromptTuning for extremely parameter-efficient
fine-tuning. Only ~50K parameters are trained (vs ~50M for LoRA r=32).

This is ideal for large models (32B+) with small datasets where LoRA is unstable.
"""

from dataclasses import asdict
from pathlib import Path
from typing import Any

from appinfra import DotDict
from appinfra.log import Logger

from ..prompt.config import Config as PromptConfig
from .base import SFTBase
from .peft_config import prompt_peft_config


class Trainer(SFTBase):
    """Trainer for prompt tuning adapters using supervised fine-tuning."""

    METHOD = "prompt"
    LABEL = "prompt tuning"

    def __init__(
        self,
        lg: Logger,
        data_path: str | Path,
        output_dir: str | Path,
        base_model: str,
        prompt_config: PromptConfig | None = None,
        training_config: DotDict | None = None,
        quantize: bool | None = None,
    ):
        super().__init__(lg, data_path, output_dir, base_model, training_config, quantize)
        self.prompt_config = prompt_config or PromptConfig()

    def _load_model(self) -> None:
        """Load base model and apply prompt tuning."""
        from peft import get_peft_model

        self._load_base_model()
        if self._is_quantized:
            from peft import prepare_model_for_kbit_training

            self.model = prepare_model_for_kbit_training(self.model)

        if self.training_config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        peft_config = prompt_peft_config(self.prompt_config, self.base_model)
        self.model = get_peft_model(self.model, peft_config)
        self._log_trainable_params(decimals=6)

    def _run_config(self) -> dict[str, Any]:
        """Hyperparameters the run records in its result."""
        return {
            "prompt_tuning": asdict(self.prompt_config),
            "training": self._training_summary(),
            "quantized": self._is_quantized,
        }
