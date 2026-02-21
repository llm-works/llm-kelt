"""Training manifest schema.

Defines the structure of training manifest files - self-contained documents
that specify everything needed to run a training job:
- Model configuration
- LoRA parameters
- Training hyperparameters
- Training data (inline or external)
- Lineage (parent adapter)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

from appinfra import DotDict


@dataclass
class Source:
    """Provenance information for the manifest.

    Attributes:
        context_key: Agent context key that created this manifest.
        description: Human-readable description of the training goal.
    """

    context_key: str | None = None
    description: str | None = None


@dataclass
class Model:
    """Base model configuration.

    Attributes:
        base: HuggingFace model ID or local path.
        quantize: Use 4-bit quantization (QLoRA).
    """

    base: str = "Qwen/Qwen2.5-7B-Instruct"
    quantize: bool = True


@dataclass
class Data:
    """Training data specification.

    Data can be inline (records embedded in manifest) or external (path to JSONL).

    Attributes:
        format: "inline" for embedded records, "external" for file path.
        records: List of training records (for inline format).
        path: Path to JSONL file, relative to manifest (for external format).
    """

    format: Literal["inline", "external"] = "inline"
    records: list[dict[str, Any]] = field(default_factory=list)
    path: str | None = None

    def __post_init__(self) -> None:
        """Validate data specification."""
        if self.format == "inline" and not self.records:
            raise ValueError("Inline data format requires non-empty records list")
        if self.format == "external" and not self.path:
            raise ValueError("External data format requires a path")


@dataclass
class Manifest:
    """Complete training manifest.

    Self-contained specification for a training job. Can be created by an agent,
    submitted to a training queue, and executed on any machine with access to
    the base model and adapter registry.

    Attributes:
        version: Schema version for forward compatibility.
        created_at: When manifest was created.
        method: Training method ("dpo" or "sft").
        adapter_id: Output adapter identifier.
        source: Provenance information.
        model: Base model configuration.
        parent_adapter: Parent adapter path or registry ID (for lineage).
        lora: LoRA configuration (converted to lora.Config by clients).
        training: Training hyperparameters (merged with TRAINING_DEFAULTS by trainers).
        method_config: Method-specific configuration (interpreted by dpo/sft clients).
        data: Training data specification.
    """

    # Required fields
    adapter_id: str
    method: Literal["dpo", "sft"]
    data: Data

    # Version and metadata
    version: int = 1
    created_at: datetime = field(default_factory=lambda: datetime.now().astimezone())

    # Optional configuration
    source: Source = field(default_factory=Source)
    model: Model = field(default_factory=Model)
    parent_adapter: str | None = None
    lora: DotDict = field(default_factory=DotDict)
    training: DotDict = field(default_factory=DotDict)
    method_config: DotDict = field(default_factory=DotDict)

    def __post_init__(self) -> None:
        """Validate manifest."""
        if not self.adapter_id:
            raise ValueError("adapter_id is required")
        if self.method not in ("dpo", "sft"):
            raise ValueError(f"method must be 'dpo' or 'sft', got '{self.method}'")
        # Validate adapter_id has no path traversal
        if "/" in self.adapter_id or "\\" in self.adapter_id or ".." in self.adapter_id:
            raise ValueError(f"Invalid adapter_id: {self.adapter_id}")
