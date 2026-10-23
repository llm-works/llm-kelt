# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-kelt Authors

"""Conversion of kelt's adapter configs to PEFT configs."""

from peft import LoraConfig as PeftLoraConfig
from peft import PromptTuningConfig as PeftPromptTuningConfig
from peft import PromptTuningInit, TaskType

from ..lora.config import Config as LoraConfig
from ..prompt.config import Config as PromptConfig

_TASK_TYPES = {
    "CAUSAL_LM": TaskType.CAUSAL_LM,
    "SEQ_CLS": TaskType.SEQ_CLS,
    "SEQ_2_SEQ_LM": TaskType.SEQ_2_SEQ_LM,
    "TOKEN_CLS": TaskType.TOKEN_CLS,
}


def lora_peft_config(config: LoraConfig) -> PeftLoraConfig:
    """Convert a LoRA config to a PEFT LoraConfig for get_peft_model().

    Raises:
        ValueError: If target_modules is unset; resolve() the config first.
    """
    if config.target_modules is None:
        raise ValueError("target_modules is unset; resolve() the config for the model first")
    return PeftLoraConfig(
        r=config.r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        exclude_modules=config.exclude_modules,
        bias=config.bias,
        task_type=_TASK_TYPES[config.task_type],
        use_rslora=config.use_rslora,
    )


def prompt_peft_config(config: PromptConfig, tokenizer_name_or_path: str) -> PeftPromptTuningConfig:
    """Convert a prompt tuning config to a PEFT PromptTuningConfig for get_peft_model().

    Args:
        config: Prompt tuning config.
        tokenizer_name_or_path: HuggingFace model ID or path for the tokenizer, used to
            tokenize the init text for TEXT initialization.
    """
    init = {"TEXT": PromptTuningInit.TEXT, "RANDOM": PromptTuningInit.RANDOM}
    kwargs: dict = {
        "task_type": _TASK_TYPES[config.task_type],
        "num_virtual_tokens": config.num_virtual_tokens,
        "prompt_tuning_init": init[config.prompt_tuning_init],
    }
    if config.prompt_tuning_init == "TEXT":
        kwargs["prompt_tuning_init_text"] = config.prompt_tuning_init_text
        kwargs["tokenizer_name_or_path"] = tokenizer_name_or_path
    return PeftPromptTuningConfig(**kwargs)
