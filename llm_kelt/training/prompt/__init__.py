# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-kelt Authors

"""Prompt tuning package.

Provides:
- Config: Prompt tuning configuration
- Trainer: Prompt tuning using TRL
- Client: High-level client for manifest-based training
"""

from typing import Any

from ..runtime import load
from .client import Client
from .config import Config

__all__ = ["Config", "Trainer", "Client"]


def __getattr__(name: str) -> Any:
    """Trainer, loaded on first use from the training runtime."""
    if name == "Trainer":
        return load("prompt").Trainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
