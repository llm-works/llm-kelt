# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-kelt Authors

"""Training modules for Kelt framework.

Contains utilities for training data preparation and model fine-tuning:
- manifest: File-based training workflow (create, submit, run manifests)
- dpo: DPO training client and trainer
- sft: SFT training client
- lora: LoRA/SFT training and adapter registry
- schema: Training dataclasses (Adapter, RunResult)
- runtime: the trainers and all code that imports the ML stack, loaded through runtime.load()

Install training dependencies with: pip install llm-kelt[training]
"""

from typing import Any

from .dpo import Client as DpoClient
from .export import ExportResult, export_feedback_classifier, export_feedback_sft
from .factory import Factory
from .lineage import Build, BuildState, SeriesInfo, SetEntry, Submission
from .lora import AdapterInfo, AdapterRegistry
from .lora import Config as LoraConfig
from .manifest import Client, Data, Manifest, Source
from .profiles import (
    ProfileDetectionError,
    build_training_config,
    get_registry_path,
    load_default_profile,
)
from .runner import Runner
from .runtime import load
from .schema import (
    TRAINING_DEFAULTS,
    ActiveSeries,
    Adapter,
    QueueStatus,
    RunResult,
    SubmitResult,
)
from .sft import Client as SftClient

__all__ = [
    # Schema
    "TRAINING_DEFAULTS",
    "Adapter",
    "QueueStatus",
    "RunResult",
    "SubmitResult",
    "ProfileDetectionError",
    "build_training_config",
    "load_default_profile",
    "get_registry_path",
    # Export functions
    "ExportResult",
    "export_feedback_sft",
    "export_feedback_classifier",
    # Adapter registry
    "AdapterRegistry",
    "AdapterInfo",
    "LoraConfig",
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
    # Lineage
    "SeriesInfo",
    "SetEntry",
    "Build",
    "BuildState",
    "Submission",
    "ActiveSeries",
    # Training functions (lazy-loaded, require 'training' extras)
    "train_lora",
    "train_dpo",
]


def __getattr__(name: str) -> Any:
    """train_lora and train_dpo, loaded on first use from the training runtime."""
    if name == "train_lora":
        return load("lora").train_lora
    if name == "train_dpo":
        return load("dpo").train_dpo
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
