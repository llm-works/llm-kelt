"""Training configuration dataclasses.

Provides generic training hyperparameters and result tracking.
Defaults are optimized for Qwen2.5-7B-Instruct with QLoRA.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class RunConfig:
    """Training hyperparameters.

    Attributes:
        num_epochs: Number of training epochs.
        batch_size: Per-device batch size.
        gradient_accumulation_steps: Steps to accumulate before update.
            Effective batch = batch_size * gradient_accumulation_steps.
        learning_rate: Peak learning rate after warmup.
        warmup_ratio: Fraction of steps for linear warmup.
        max_seq_length: Maximum sequence length (truncates longer sequences).
        logging_steps: Log metrics every N steps.
        save_steps: Save checkpoint every N steps.
        eval_split: Fraction of data to use for evaluation (0 = no eval).
        fp16: Use FP16 mixed precision training.
        bf16: Use BF16 mixed precision (preferred on Ampere+ GPUs).
        gradient_checkpointing: Trade compute for memory.
        seed: Random seed for reproducibility.
    """

    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    max_seq_length: int = 2048
    logging_steps: int = 10
    save_steps: int = 100
    eval_split: float = 0.0
    fp16: bool = True
    bf16: bool = False
    gradient_checkpointing: bool = True
    seed: int = 42

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.num_epochs <= 0:
            raise ValueError(f"num_epochs must be positive, got {self.num_epochs}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError(
                f"gradient_accumulation_steps must be positive, "
                f"got {self.gradient_accumulation_steps}"
            )
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if not 0.0 <= self.warmup_ratio < 1.0:
            raise ValueError(f"warmup_ratio must be in [0, 1), got {self.warmup_ratio}")
        if self.max_seq_length <= 0:
            raise ValueError(f"max_seq_length must be positive, got {self.max_seq_length}")
        if not 0.0 <= self.eval_split < 1.0:
            raise ValueError(f"eval_split must be in [0, 1), got {self.eval_split}")
        if self.fp16 and self.bf16:
            raise ValueError("Cannot enable both fp16 and bf16")

    @property
    def effective_batch_size(self) -> int:
        """Calculate effective batch size including gradient accumulation."""
        return self.batch_size * self.gradient_accumulation_steps


@dataclass
class RunResult:
    """Result of a training run.

    Attributes:
        adapter_path: Path to saved adapter weights.
        base_model: HuggingFace model ID used as base.
        method: Training method ("lora" or "dpo").
        metrics: Training metrics (loss, eval metrics if applicable).
        config: Full configuration dict used for training.
        started_at: When training started.
        completed_at: When training completed.
        samples_trained: Total number of samples seen during training.
        based_on: Parent adapter this was trained from (for lineage tracking).
    """

    adapter_path: Path
    base_model: str
    method: str
    metrics: dict
    config: dict
    started_at: datetime
    completed_at: datetime
    samples_trained: int
    based_on: Path | None = None

    @property
    def duration_seconds(self) -> float:
        """Calculate training duration in seconds."""
        return (self.completed_at - self.started_at).total_seconds()
