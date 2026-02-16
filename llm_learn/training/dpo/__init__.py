"""DPO (Direct Preference Optimization) training package.

Provides:
- Client: Training run management with pair assignment
- Trainer: Actual DPO training using TRL
- train_dpo: Convenience function for training
"""

from .client import (
    Client,
    PairTuple,
    PendingPair,
    TrainedPair,
    TrainingRun,
    TrainingRunInfo,
)

__all__ = [
    # Core client
    "Client",
    # Models
    "TrainingRun",
    "TrainingRunInfo",
    "PendingPair",
    "TrainedPair",
    # Type aliases
    "PairTuple",
    # Training functions (lazy-loaded, require 'training' extras)
    "Trainer",
    "train_dpo",
]


def __getattr__(name: str):
    """Lazy import training classes that require optional dependencies."""
    if name == "Trainer":
        from .trainer import Trainer

        return Trainer
    if name == "train_dpo":
        from .trainer import train_dpo

        return train_dpo
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
