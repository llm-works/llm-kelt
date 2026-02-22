"""Training modules for Learn framework.

Contains utilities for training data preparation and model fine-tuning:
- manifest: File-based training workflow (create, submit, run manifests)
- dpo: DPO training client and trainer
- sft: SFT training client
- lora: LoRA/SFT training and adapter registry
- schema: Training dataclasses (Adapter, RunResult)

Install training dependencies with: pip install llm-learn[training]
"""

from .dpo import Client as DpoClient
from .export import ExportResult, export_feedback_classifier, export_feedback_sft
from .factory import Factory
from .lora import AdapterInfo, AdapterRegistry
from .manifest import Client, Data, Manifest, Source
from .profiles import build_training_config, get_registry_path, load_default_profile, load_profile
from .runner import Runner
from .schema import TRAINING_DEFAULTS, Adapter, RunResult
from .sft import Client as SftClient

__all__ = [
    # Schema
    "TRAINING_DEFAULTS",
    "Adapter",
    "RunResult",
    "build_training_config",
    "load_default_profile",
    "load_profile",  # deprecated, use load_default_profile
    "get_registry_path",
    # Export functions
    "ExportResult",
    "export_feedback_sft",
    "export_feedback_classifier",
    # Adapter registry
    "AdapterRegistry",
    "AdapterInfo",
    # Clients
    "Factory",
    "Client",
    "DpoClient",
    "SftClient",
    # Manifest
    "Runner",
    "Manifest",
    "Source",
    "Data",
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
