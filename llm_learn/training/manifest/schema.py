"""Training manifest schema.

Defines the structure of training manifest files - self-contained documents
that specify everything needed to run a training job:
- Model configuration
- LoRA parameters
- Training hyperparameters
- Training data (inline or external)
- Lineage (parent adapter)
- Training output (populated after completion)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

from appinfra import DotDict

from ..schema import Adapter, RunResult


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
        adapter: Output adapter key (series name, e.g., "my-agent-sft").
        method: Training method ("dpo" or "sft").
        data: Training data specification.
        version: Schema version for forward compatibility.
        created_at: When manifest was created.
        source: Provenance information.
        parent: Parent adapter for lineage (continue training from this adapter).
        lora: LoRA configuration (converted to lora.Config by clients).
        training: Training config including:
            - requested_model: Optional model requested by agent (resolved at training time).
            - num_epochs, learning_rate, etc.: Training hyperparameters.
            - output: Training result (populated after completion).
        method_config: Method-specific configuration (interpreted by dpo/sft clients).
    """

    # Required fields
    adapter: str
    method: Literal["dpo", "sft"]
    data: Data

    # Version and metadata
    version: int = 1
    created_at: datetime = field(default_factory=lambda: datetime.now().astimezone())

    # Optional configuration
    source: Source = field(default_factory=Source)
    parent: Adapter | None = None
    lora: DotDict = field(default_factory=DotDict)
    training: DotDict = field(default_factory=DotDict)
    method_config: DotDict = field(default_factory=DotDict)

    # Output (populated after training)
    output: RunResult | None = None

    def __post_init__(self) -> None:
        """Validate manifest."""
        if not self.adapter:
            raise ValueError("adapter is required")
        if self.method not in ("dpo", "sft"):
            raise ValueError(f"method must be 'dpo' or 'sft', got '{self.method}'")
        # Validate adapter key has no path traversal (conservative check - ".." anywhere is rejected)
        if "/" in self.adapter or "\\" in self.adapter or ".." in self.adapter:
            raise ValueError(f"Invalid adapter: {self.adapter}")
