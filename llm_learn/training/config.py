"""Training configuration.

Provides training hyperparameters and result tracking.
Defaults are optimized for Qwen2.5-7B-Instruct with QLoRA.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from appinfra import DotDict

# Training hyperparameter defaults
TRAINING_DEFAULTS = DotDict(
    num_epochs=3,
    batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    warmup_ratio=0.03,
    max_seq_length=2048,
    logging_steps=10,
    save_steps=100,
    eval_split=0.0,
    fp16=True,
    bf16=False,
    gradient_checkpointing=True,
    seed=42,
)


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
