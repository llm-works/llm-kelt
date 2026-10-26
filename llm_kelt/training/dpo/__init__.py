# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-kelt Authors

"""DPO (Direct Preference Optimization) training package.

Provides:
- Client: Training client that operates on manifests
- Trainer: Actual DPO training using TRL
- train_dpo: Convenience function for training
- export_preferences: Export atomic preferences to DPO format
"""

from typing import Any

from ..runtime import load
from .client import Client
from .export import ExportResult, export_preferences

__all__ = [
    "Client",
    "ExportResult",
    "export_preferences",
    "Trainer",
    "train_dpo",
]


def __getattr__(name: str) -> Any:
    """Trainer and train_dpo, loaded on first use from the training runtime."""
    if name in ("Trainer", "train_dpo"):
        return getattr(load("dpo"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
