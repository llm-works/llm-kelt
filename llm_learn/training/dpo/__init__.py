"""DPO (Direct Preference Optimization) training package.

Provides:
- DpoClient: Training run management with pair assignment
- DpoTrainer: Actual DPO training using TRL
- train_dpo: Convenience function for training
"""

from .client import (
    DpoClient,
    DpoPendingPair,
    DpoRun,
    DpoRunInfo,
    DpoRunPair,
    DpoTrainedPair,
    PairTuple,
    TrainingRun,
    TrainingRunInfo,
)

__all__ = [
    # Core client
    "DpoClient",
    # Models
    "TrainingRun",
    "TrainingRunInfo",
    "DpoPendingPair",
    "DpoTrainedPair",
    # Backward compatibility aliases
    "DpoRun",
    "DpoRunInfo",
    "DpoRunPair",
    # Type aliases
    "PairTuple",
    # Training functions (lazy-loaded, require 'training' extras)
    "DpoTrainer",
    "train_dpo",
]


def __getattr__(name: str):
    """Lazy import training classes that require optional dependencies."""
    if name == "DpoTrainer":
        from .trainer import DpoTrainer

        return DpoTrainer
    if name == "train_dpo":
        from .trainer import train_dpo

        return train_dpo
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
