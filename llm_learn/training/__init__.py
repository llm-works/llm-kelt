"""Training modules for Learn framework.

Contains utilities for training data preparation and model fine-tuning:
- export: Data export to training formats (DPO, SFT, classifier)
- config: Training configuration dataclasses
- runs: Training run tracking for iterative workflows
- lora: LoRA/SFT training (requires 'training' extras)
- dpo: DPO training (requires 'training' extras)

Install training dependencies with: pip install llm-learn[training]
"""

from .config import LoraConfig, TrainingConfig, TrainingResult
from .export import (
    ExportResult,
    export_feedback_classifier,
    export_feedback_sft,
    export_preferences_dpo,
)
from .registry import AdapterInfo, AdapterRegistry
from .runs import TrainingRun, TrainingRunClient, TrainingRunInfo, TrainingRunPair

__all__ = [
    # Export functions
    "ExportResult",
    "export_preferences_dpo",
    "export_feedback_sft",
    "export_feedback_classifier",
    # Config dataclasses
    "LoraConfig",
    "TrainingConfig",
    "TrainingResult",
    # Adapter registry
    "AdapterRegistry",
    "AdapterInfo",
    # Training run tracking
    "TrainingRun",
    "TrainingRunClient",
    "TrainingRunInfo",
    "TrainingRunPair",
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
