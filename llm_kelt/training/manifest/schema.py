# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-kelt Authors

"""Training manifest schema.

Defines the structure of training manifest files - self-contained documents
that specify everything needed to run a training job.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from appinfra import DotDict, FieldDict, field

if TYPE_CHECKING:
    from ..schema import Adapter, RunResult


class Source(FieldDict):
    """Provenance information for the manifest."""

    # Agent context key that created this manifest
    context_key: str | None = None
    # Database schema where training data originated (for workspace isolation)
    schema_name: str | None = None
    # Human-readable description of the training goal
    description: str | None = None


class Data(FieldDict):
    """Training data specification (inline or external file)."""

    # "inline" for embedded records, "external" for file path
    format: Literal["inline", "external"]
    # List of training records (for inline format)
    records: list[dict[str, Any]] = field(default_factory=list)
    # Path to JSONL file, relative to manifest (for external format)
    path: str | None = None
    # Shared prompt templates for SFT records, keyed by template id (see templates.py).
    # Applies to inline and external records.
    templates: dict[str, str] = field(default_factory=dict)


class Deployment(FieldDict):
    """Deployment configuration after training."""

    # "skip" (don't deploy), "add" (keep existing), or "replace" (remove existing)
    policy: Literal["skip", "add", "replace"] = "replace"


class Manifest(FieldDict):
    """Complete training manifest."""

    # Output adapter key (series name, e.g., "my-agent-sft")
    adapter: str
    # Training method
    method: Literal["dpo", "sft", "prompt"]
    # Training data specification
    data: Data
    # Series of the adapter this manifest's data is a new set of (see training/lineage.py).
    # Unset: a one-off training on this data alone.
    series: str | None = None
    # Lineage commit a series build trains, set when the build is claimed
    commit: str | None = None
    # Annotations of this manifest's set (a small JSON object, series manifests only)
    annotations: dict[str, Any] = field(default_factory=dict)
    # Deployment configuration
    deployment: Deployment | None = None
    # Schema version for forward compatibility
    version: str = "1"
    # When manifest was created
    created_at: datetime = field(default_factory=datetime.now)
    # Provenance information
    source: Source | None = None
    # Parent adapter for lineage (continue training from this)
    parent: Adapter | None = None
    # LoRA configuration
    lora: DotDict = field(default_factory=DotDict)
    # Training config (num_epochs, learning_rate, requested_model, etc.)
    training: DotDict = field(default_factory=DotDict)
    # Method-specific configuration (beta for DPO, etc.)
    method_config: DotDict = field(default_factory=DotDict)
    # Training result, populated after completion
    output: RunResult | None = None
    # Path to manifest file (set during loading, not serialized)
    source_path: Path | None = None


def get_deploy_setting(manifest: Manifest) -> bool | Literal["add", "replace"]:
    """Get deployment setting from manifest.

    Args:
        manifest: Training manifest.

    Returns:
        False if policy is "skip", otherwise "add" or "replace".
    """
    policy = manifest.deployment.policy if manifest.deployment else "replace"
    if policy == "skip":
        return False
    return policy
