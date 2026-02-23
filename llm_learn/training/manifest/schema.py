"""Training manifest schema.

Defines the structure of training manifest files - self-contained documents
that specify everything needed to run a training job.
"""

from __future__ import annotations

from appinfra import DotDict


class Source(DotDict):
    """Provenance information for the manifest.

    Fields:
        context_key: Agent context key that created this manifest.
        description: Human-readable description of the training goal.
    """

    pass


class Data(DotDict):
    """Training data specification.

    Data can be inline (records embedded in manifest) or external (path to JSONL).

    Fields:
        format: "inline" for embedded records, "external" for file path.
        records: List of training records (for inline format).
        path: Path to JSONL file, relative to manifest (for external format).
    """

    pass


class Deployment(DotDict):
    """Deployment configuration.

    Controls how the trained adapter is deployed after training.

    Fields:
        policy: Deployment policy - "skip" (don't deploy), "add" (deploy as new
            version keeping existing), or "replace" (deploy and remove existing).
            Default is "replace".
    """

    pass


class Manifest(DotDict):
    """Complete training manifest.

    Self-contained specification for a training job. Can be created by an agent,
    submitted to a training queue, and executed on any machine with access to
    the base model and adapter registry.

    Fields:
        adapter: Output adapter key (series name, e.g., "my-agent-sft").
        method: Training method ("dpo" or "sft").
        data: Training data specification (Data).
        deployment: Deployment configuration (Deployment).
        version: Schema version for forward compatibility.
        created_at: When manifest was created (datetime).
        source: Provenance information (Source).
        parent: Parent adapter for lineage (Adapter).
        lora: LoRA configuration.
        training: Training config (num_epochs, learning_rate, requested_model, etc.).
        method_config: Method-specific configuration (beta for DPO, etc.).
        output: Training result, populated after completion (RunResult).
    """

    pass
