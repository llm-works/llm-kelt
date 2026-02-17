"""LoRA adapter configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from peft import LoraConfig as PeftLoraConfig

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


@dataclass
class Config:
    """LoRA adapter configuration.

    Attributes:
        r: LoRA rank. Higher = more capacity but more VRAM. 16 is a good default.
        lora_alpha: Scaling factor. Usually 2x rank.
        lora_dropout: Dropout for LoRA layers. 0.05 prevents overfitting.
        target_modules: Which modules to apply LoRA to. Defaults to Qwen2.5 attention + MLP.
        bias: Bias training mode. "none" is most memory efficient.
        task_type: Task type for PEFT. "CAUSAL_LM" for language modeling.
    """

    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: QWEN_TARGET_MODULES.copy())
    bias: Literal["none", "all", "lora_only"] = "none"
    task_type: Literal["CAUSAL_LM", "SEQ_CLS", "SEQ_2_SEQ_LM", "TOKEN_CLS"] = "CAUSAL_LM"

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.r <= 0:
            raise ValueError(f"LoRA rank must be positive, got {self.r}")
        if self.lora_alpha <= 0:
            raise ValueError(f"LoRA alpha must be positive, got {self.lora_alpha}")
        if not 0.0 <= self.lora_dropout < 1.0:
            raise ValueError(f"LoRA dropout must be in [0, 1), got {self.lora_dropout}")
        if not self.target_modules:
            raise ValueError("target_modules cannot be empty")
        if self.bias not in ("none", "all", "lora_only"):
            raise ValueError(f"bias must be 'none', 'all', or 'lora_only', got {self.bias}")
        if self.task_type not in ("CAUSAL_LM", "SEQ_CLS", "SEQ_2_SEQ_LM", "TOKEN_CLS"):
            raise ValueError(f"Invalid task_type: {self.task_type}")

    def to_peft_config(self) -> PeftLoraConfig:
        """Convert to PEFT LoraConfig for training.

        Returns:
            peft.LoraConfig instance ready for get_peft_model()
        """
        from peft import LoraConfig as PeftLoraConfig
        from peft import TaskType

        task_type_map = {
            "CAUSAL_LM": TaskType.CAUSAL_LM,
            "SEQ_CLS": TaskType.SEQ_CLS,
            "SEQ_2_SEQ_LM": TaskType.SEQ_2_SEQ_LM,
            "TOKEN_CLS": TaskType.TOKEN_CLS,
        }

        return PeftLoraConfig(
            r=self.r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=self.target_modules,
            bias=self.bias,
            task_type=task_type_map[self.task_type],
        )
