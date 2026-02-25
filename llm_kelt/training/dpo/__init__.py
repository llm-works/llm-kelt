"""DPO (Direct Preference Optimization) training package.

Provides:
- Client: Training client that operates on manifests
- Trainer: Actual DPO training using TRL
- train_dpo: Convenience function for training
- export_preferences: Export atomic preferences to DPO format
"""

from .client import Client
from .export import ExportResult, export_preferences

__all__ = [
    "Client",
    "ExportResult",
    "export_preferences",
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
