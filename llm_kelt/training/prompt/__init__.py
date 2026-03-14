"""Prompt tuning package.

Provides:
- Config: Prompt tuning configuration
- Trainer: Prompt tuning using TRL
- Client: High-level client for manifest-based training
"""

from .client import Client
from .config import Config

__all__ = ["Config", "Trainer", "Client"]


def __getattr__(name: str):
    """Lazy import training classes that require optional dependencies."""
    if name == "Trainer":
        from .trainer import Trainer

        return Trainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
