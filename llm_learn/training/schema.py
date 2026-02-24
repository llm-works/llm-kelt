"""Training schema.

Provides training hyperparameters and result tracking.
Defaults are optimized for Qwen2.5-7B-Instruct with QLoRA.
"""

from __future__ import annotations

from appinfra import DotDict


class AdapterInfo(DotDict):
    """Information about a registered adapter.

    Fields:
        key: Adapter key (unique identifier).
        path: Path to adapter directory.
        deployed: Whether adapter is currently deployed.
        version_id: Version identifier (YYYYMMDD-HHMMSS-md5).
        description: Human-readable description.
        md5: MD5 hash of adapter weights.
        parent: Parent adapter md5 (for lineage tracking).
    """

    pass


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


class Adapter(DotDict):
    """Adapter identity.

    Fields:
        md5: MD5 hash of weights (12 char hex). THE unique version identifier.
        mtime: ISO timestamp of weights file modification time.
        path: Full path to adapter directory.
    """

    pass


class RunResult(DotDict):
    """Result of a training run.

    Fields:
        status: "completed" or "failed".
        base_model: HuggingFace model ID used as base.
        method: Training method ("sft" or "dpo").
        metrics: Training metrics (loss, eval metrics if applicable).
        config: Full configuration dict used for training.
        started_at: When training started (datetime).
        completed_at: When training completed (datetime).
        samples_trained: Total number of samples seen during training.
        adapter: Adapter identity with md5/mtime/path (None if failed).
        parent: Parent adapter this was trained from (for lineage tracking).
        error: Error message if status is "failed".
    """

    @property
    def duration_seconds(self) -> float:
        """Calculate training duration in seconds."""
        return float((self.completed_at - self.started_at).total_seconds())
