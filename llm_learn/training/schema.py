"""Training schema.

Provides training hyperparameters and result tracking.
Defaults are optimized for Qwen2.5-7B-Instruct with QLoRA.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from appinfra import DataDotDict, DotDict, field


class AdapterInfo(DataDotDict):
    """Information about a registered adapter."""

    # Adapter key (unique identifier)
    key: str
    # Path to adapter directory
    path: Path
    # Version identifier (YYYYMMDD-HHMMSS-md5)
    version_id: str
    # MD5 hash of adapter weights
    md5: str
    # Whether adapter is currently deployed
    deployed: bool = False
    # Human-readable description
    description: str = ""
    # Parent adapter md5 (for lineage tracking)
    parent: str | None = None


class SubmitResult(DataDotDict):
    """Result of submitting a manifest."""

    # Adapter key
    adapter: str
    # When submitted
    timestamp: datetime
    # Storage location (path for file, URI for DB, etc.)
    location: str


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


class Adapter(DataDotDict):
    """Adapter identity."""

    # MD5 hash of weights (12 char hex) - THE unique version identifier
    md5: str
    # ISO timestamp of weights file modification time
    mtime: str
    # Full path to adapter directory
    path: str


class RunResult(DataDotDict):
    """Result of a training run."""

    # "completed" or "failed"
    status: str | None = None
    # When training started
    started_at: datetime
    # When training completed
    completed_at: datetime
    # HuggingFace model ID used as base
    base_model: str | None = None
    # Training method ("sft" or "dpo")
    method: str | None = None
    # Training metrics (loss, eval metrics if applicable)
    metrics: DotDict = field(default_factory=DotDict)
    # Full configuration dict used for training
    config: DotDict = field(default_factory=DotDict)
    # Total number of samples seen during training
    samples_trained: int = 0
    # Adapter identity with md5/mtime/path (None if failed)
    adapter: Adapter | None = None
    # Parent adapter this was trained from (for lineage tracking)
    parent: Adapter | None = None
    # Error message if status is "failed"
    error: str | None = None

    @property
    def duration_seconds(self) -> float:
        """Calculate training duration in seconds."""
        return float((self.completed_at - self.started_at).total_seconds())
