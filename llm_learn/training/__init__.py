"""Training modules for Learn framework.

Contains utilities for training data preparation and model fine-tuning:
- dpo: DPO training with pair management
- lora: LoRA/SFT training and adapter registry
- export: Data export to training formats
- config: Training configuration dataclasses

Install training dependencies with: pip install llm-learn[training]
"""

from .config import RunConfig, RunResult
from .dpo import (
    PairTuple,
    TrainingRun,
    TrainingRunInfo,
)
from .export import (
    ExportResult,
    export_feedback_classifier,
    export_feedback_sft,
    export_preferences_dpo,
    generate_dpo_pairs,
)
from .lora import AdapterInfo, AdapterRegistry
from .client import Client as TrainClient

__all__ = [
    # DPO training - models (access via training.dpo.*)
    "TrainingRun",
    "TrainingRunInfo",
    "PairTuple",
    # Export functions
    "ExportResult",
    "export_preferences_dpo",
    "export_feedback_sft",
    "export_feedback_classifier",
    "generate_dpo_pairs",
    # Config dataclasses
    "RunConfig",
    "RunResult",
    # Adapter registry
    "AdapterRegistry",
    "AdapterInfo",
    # Client aggregator
    "TrainClient",
    # Training functions (lazy-loaded, require 'training' extras)
    "train_lora",
    "train_dpo",
]


def __getattr__(name: str):
    """Lazy import training functions that require optional dependencies."""
    if name == "train_lora":
        from .lora import train_lora

        return train_lora
    if name == "train_dpo":
        from .dpo import train_dpo

        return train_dpo
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
