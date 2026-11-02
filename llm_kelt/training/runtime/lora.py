# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-kelt Authors

"""LoRA training using supervised fine-tuning (SFT).

Uses TRL's SFTTrainer with PEFT for parameter-efficient fine-tuning.
Supports QLoRA (4-bit quantization) to reduce VRAM requirements.
"""

from pathlib import Path
from typing import Any

from appinfra import DotDict
from appinfra.log import Logger

from ..lora.config import Config as LoraConfig
from ..schema import RunResult
from .base import SFTBase, lora_config_summary
from .peft_config import lora_peft_config


class Trainer(SFTBase):
    """Trainer for LoRA adapters using supervised fine-tuning."""

    METHOD = "sft"
    LABEL = "LoRA training"

    def __init__(
        self,
        lg: Logger,
        data_path: str | Path,
        output_dir: str | Path,
        base_model: str,
        lora_config: LoraConfig | None = None,
        training_config: DotDict | None = None,
        quantize: bool | None = None,
    ):
        super().__init__(lg, data_path, output_dir, base_model, training_config, quantize)
        self.lora_config = lora_config or LoraConfig()

    def _apply_lora(self) -> None:
        """Apply LoRA adapter to the loaded model."""
        from peft import get_peft_model

        self.lora_config = self.lora_config.resolve(self.model.config.model_type)
        peft_config = lora_peft_config(self.lora_config)
        self.model = get_peft_model(self.model, peft_config)
        self._log_trainable_params()

    def _load_model(self) -> None:
        """Load base model with auto-detected quantization and apply LoRA."""
        from peft import prepare_model_for_kbit_training

        self._load_base_model()
        if self.training_config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        if self._is_quantized:
            self.model = prepare_model_for_kbit_training(self.model)

        self._apply_lora()

    def _run_config(self) -> dict[str, Any]:
        """Hyperparameters the run records in its result."""
        return {
            "lora": lora_config_summary(self.lora_config),
            "training": self._training_summary(),
            "quantized": self._is_quantized,
        }


def train_lora(
    lg: Logger,
    data_path: str | Path,
    output_dir: str | Path,
    base_model: str,
    lora_config: LoraConfig | None = None,
    training_config: DotDict | None = None,
    quantize: bool | None = None,
    resume_from: str | Path | None = None,
) -> RunResult:
    """Train a LoRA adapter using supervised fine-tuning.

    Args:
        lg: Logger instance.
        data_path: Path to JSONL file with {"instruction", "output", "input"?} records
            ("prompt" and "response" are accepted in place of "instruction" and "output").
        output_dir: Directory to save the trained adapter.
        base_model: HuggingFace model ID or local path.
        lora_config: LoRA hyperparameters. Uses sensible defaults if not provided.
        training_config: Training hyperparameters. Uses sensible defaults if not provided.
        quantize: Force quantization on/off. None = auto-detect from model metadata.
        resume_from: Path to checkpoint to resume training from.

    Returns:
        RunResult with adapter path, metrics, and training metadata.
    """
    trainer = Trainer(
        lg=lg,
        data_path=data_path,
        output_dir=output_dir,
        base_model=base_model,
        lora_config=lora_config,
        training_config=training_config,
        quantize=quantize,
    )
    return trainer.train(resume_from=resume_from)
