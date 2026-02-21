"""Manifest client - lifecycle management for training manifests.

Handles creating, loading, saving, and submitting manifests to the training queue.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from appinfra import DotDict
from appinfra.log import Logger

from .loader import load_manifest as _load_manifest
from .loader import save_manifest as _save_manifest
from .schema import (
    Data,
    Manifest,
    Model,
    Source,
)

if TYPE_CHECKING:
    pass

# Keys that belong in training config (vs method_config or lora)
_TRAINING_KEYS = frozenset(
    [
        "num_epochs",
        "batch_size",
        "gradient_accumulation_steps",
        "learning_rate",
        "warmup_ratio",
        "max_seq_length",
        "logging_steps",
        "save_steps",
        "eval_split",
        "fp16",
        "bf16",
        "gradient_checkpointing",
        "seed",
    ]
)


class Client:
    """Client for training manifest lifecycle management.

    Provides methods to create, load, save, and submit manifests.
    Accessed via LearnClient.train.manifest.

    Usage:
        # Create a manifest
        manifest = learn.train.manifest.create(
            adapter_id="coding-v1",
            method="dpo",
            model="Qwen/Qwen2.5-7B-Instruct",
            data=[{"prompt": "...", "chosen": "...", "rejected": "..."}],
        )

        # Save to file
        learn.train.manifest.save(manifest, Path("coding-v1.yaml"))

        # Submit to training queue
        learn.train.manifest.submit(manifest)

        # List pending manifests
        learn.train.manifest.list_pending()
    """

    def __init__(
        self,
        lg: Logger,
        registry_path: Path,
    ) -> None:
        """Initialize manifest client.

        Args:
            lg: Logger instance.
            registry_path: Base path for adapter registry and manifest queues.
        """
        self._lg = lg
        self._registry_path = Path(registry_path).expanduser()

        # Ensure directory structure exists
        for subdir in ("pending", "completed", "adapters", "deployed"):
            (self._registry_path / subdir).mkdir(parents=True, exist_ok=True)

    def _build_manifest_configs(
        self,
        method: Literal["dpo", "sft"],
        model: str,
        config: dict[str, Any] | None,
    ) -> tuple[Model, DotDict, DotDict, DotDict]:
        """Build configuration objects with optional overrides."""
        model_config = Model(base=model, quantize=config.get("quantize", True) if config else True)
        training_config = DotDict(
            {k: config[k] for k in _TRAINING_KEYS if k in config} if config else {}
        )
        lora_config = DotDict(config.get("lora", {})) if config else DotDict()
        method_config = DotDict()
        if config and method == "dpo":
            method_config = DotDict(self._extract_method_config(config, ["beta", "reference_free"]))

        return model_config, lora_config, training_config, method_config

    def create(
        self,
        adapter_id: str,
        method: Literal["dpo", "sft"],
        model: str,
        data: list[dict[str, Any]],
        *,
        parent_adapter: str | None = None,
        context_key: str | None = None,
        description: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> Manifest:
        """Create a new training manifest with inline data.

        Args:
            adapter_id: Unique identifier for the output adapter.
            method: Training method ("dpo" or "sft").
            model: Base model path or HuggingFace ID.
            data: List of training records (inline data).
            parent_adapter: Parent adapter path or registry ID for lineage.
            context_key: Agent context key for provenance.
            description: Human-readable description.
            config: Override training configuration (num_epochs, etc.).

        Returns:
            Manifest instance.
        """
        source = Source(context_key=context_key, description=description)
        model_config, lora_config, training_config, method_config = self._build_manifest_configs(
            method, model, config
        )

        manifest = Manifest(
            adapter_id=adapter_id,
            method=method,
            data=Data(format="inline", records=data),
            source=source,
            model=model_config,
            parent_adapter=parent_adapter,
            lora=lora_config,
            training=training_config,
            method_config=method_config,
        )

        self._lg.info("created manifest", extra={"adapter_id": adapter_id, "method": method})
        return manifest

    def _extract_method_config(self, overrides: dict[str, Any], keys: list[str]) -> dict[str, Any]:
        """Extract method-specific config values from overrides."""
        return {k: overrides[k] for k in keys if k in overrides}

    def load(self, path: Path) -> Manifest:
        """Load a manifest from file.

        Args:
            path: Path to manifest YAML file.

        Returns:
            Manifest instance.
        """
        return _load_manifest(path)

    def save(self, manifest: Manifest, path: Path) -> None:
        """Save a manifest to file.

        Args:
            manifest: Manifest to save.
            path: Output path for YAML file.
        """
        _save_manifest(manifest, path)
        self._lg.info("saved manifest", extra={"path": str(path)})

    def submit(self, manifest: Manifest) -> Path:
        """Submit a manifest to the pending training queue.

        Saves the manifest to the registry's pending directory.

        Args:
            manifest: Manifest to submit.

        Returns:
            Path to manifest in pending queue.

        Raises:
            ValueError: If manifest with same adapter_id already pending.
        """
        pending_dir = self._registry_path / "pending"
        dest_path = pending_dir / f"{manifest.adapter_id}.yaml"

        if dest_path.exists():
            raise ValueError(f"Manifest already in queue: {manifest.adapter_id}")

        _save_manifest(manifest, dest_path)
        self._lg.info("submitted manifest", extra={"path": str(dest_path)})

        return dest_path

    def list_pending(self) -> list[Path]:
        """List manifests waiting for training.

        Returns:
            List of paths to pending manifest files.
        """
        pending_dir = self._registry_path / "pending"
        return sorted(pending_dir.glob("*.yaml"))

    def list_completed(self) -> list[Path]:
        """List completed manifests.

        Returns:
            List of paths to completed manifest files.
        """
        completed_dir = self._registry_path / "completed"
        return sorted(completed_dir.glob("*.yaml"))

    def get_pending(self, adapter_id: str) -> Manifest | None:
        """Get a pending manifest by adapter ID.

        Args:
            adapter_id: Adapter ID to look up.

        Returns:
            Manifest or None if not found.
        """
        path = self._registry_path / "pending" / f"{adapter_id}.yaml"
        if not path.exists():
            return None
        return _load_manifest(path)

    def remove_pending(self, adapter_id: str) -> None:
        """Remove a manifest from the pending queue.

        Args:
            adapter_id: Adapter ID of manifest to remove.

        Raises:
            FileNotFoundError: If manifest not in queue.
        """
        pending_path = self._registry_path / "pending" / f"{adapter_id}.yaml"
        if not pending_path.exists():
            raise FileNotFoundError(f"Manifest not in queue: {adapter_id}")
        pending_path.unlink()
        self._lg.info("removed from queue", extra={"adapter_id": adapter_id})
