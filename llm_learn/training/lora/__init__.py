"""LoRA/SFT training package.

Provides:
- LoraTrainer: LoRA training using TRL
- train_lora: Convenience function for training
- AdapterRegistry: Adapter registration with llm-infer
"""

from .registry import AdapterInfo, AdapterRegistry

__all__ = [
    "AdapterInfo",
    "AdapterRegistry",
    "LoraTrainer",
    "train_lora",
]


def __getattr__(name: str):
    """Lazy import training classes that require optional dependencies."""
    if name == "LoraTrainer":
        from .trainer import LoraTrainer

        return LoraTrainer
    if name == "train_lora":
        from .trainer import train_lora

        return train_lora
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
