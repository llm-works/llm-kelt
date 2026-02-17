"""DPO (Direct Preference Optimization) training package.

Provides:
- Client: Training run management with pair assignment
- Trainer: Actual DPO training using TRL
- train_dpo: Convenience function for training
- export_run_pairs: Export pending pairs for a run to TRL DPO format
- export_preferences: Export from atomic preferences (legacy)
- generate_pairs: Generate pairs for Client.assign_pairs()
"""

from .client import (
    Client,
    PendingPair,
    Run,
    RunInfo,
    TrainedPair,
    _not_deleted_filter,
)
from .export import PairTuple, export_preferences, export_run_pairs, generate_pairs

__all__ = [
    # Core client
    "Client",
    "_not_deleted_filter",
    # Models
    "Run",
    "RunInfo",
    "PendingPair",
    "TrainedPair",
    # Export
    "PairTuple",
    "export_preferences",
    "export_run_pairs",
    "generate_pairs",
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
