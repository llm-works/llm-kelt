"""LoRA/SFT training package.

Provides:
- Config: LoRA adapter configuration
- Trainer: LoRA training using TRL
- train_lora: Convenience function for training
- AdapterRegistry: Adapter registration with llm-infer
"""

from ..schema import AdapterInfo
from .config import Config
from .registry import AdapterRegistry

__all__ = [
    "Config",
    "AdapterInfo",
    "AdapterRegistry",
    "Trainer",
    "train_lora",
]


def __getattr__(name: str):
    """Lazy import training classes that require optional dependencies."""
    if name == "Trainer":
        from .trainer import Trainer

        return Trainer
    if name == "train_lora":
        from .trainer import train_lora

        return train_lora
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
