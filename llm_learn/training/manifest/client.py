"""Manifest client - lifecycle management for training manifests.

Handles creating, loading, saving, and submitting manifests to the training queue.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from appinfra import DotDict
from appinfra.log import Logger

from ..schema import Adapter
from ..storage import Storage
from .loader import load_manifest as _load_manifest
from .loader import save_manifest as _save_manifest
from .schema import Data, Deployment, Manifest, Source

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

# Map profile keys to training config keys
_PROFILE_KEY_MAP = {
    "epochs": "num_epochs",
}


class Client:
    """Client for training manifest lifecycle management.

    Provides methods to create, load, save, and submit manifests.
    Accessed via LearnClient.train.manifest.

    Usage:
        # Create a manifest
        manifest = learn.train.manifest.create(
            adapter="coding-v1",
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
        storage: Storage,
        default_profiles: dict[str, dict] | None = None,
    ) -> None:
        """Initialize manifest client.

        Args:
            lg: Logger instance.
            storage: Storage instance for filesystem operations.
            default_profiles: Default training profiles by method.
        """
        self._lg = lg
        self._storage = storage
        self._default_profiles = default_profiles or {}

    def _build_manifest_configs(
        self,
        method: Literal["dpo", "sft"],
        model: str | None,
        config: dict[str, Any] | None,
    ) -> tuple[DotDict, DotDict, DotDict]:
        """Build configuration objects by merging default profile with agent overrides."""
        # Start with default profile for this method
        defaults = dict(self._default_profiles.get(method, {}))
        # Agent config overrides defaults
        merged = {**defaults, **(config or {})}

        # Map profile keys to training config keys (e.g., epochs -> num_epochs)
        for profile_key, config_key in _PROFILE_KEY_MAP.items():
            if profile_key in merged and config_key not in merged:
                merged[config_key] = merged.pop(profile_key)

        # Build training config with model info
        training_config = DotDict({k: merged[k] for k in _TRAINING_KEYS if k in merged})
        if model:
            training_config["requested_model"] = model

        lora_config = DotDict(merged.get("lora", {}))
        method_config = DotDict()
        if method == "dpo":
            method_config = DotDict(self._extract_method_config(merged, ["beta", "reference_free"]))

        return lora_config, training_config, method_config

    def create(
        self,
        adapter: str,
        method: Literal["dpo", "sft"],
        data: list[dict[str, Any]],
        *,
        model: str | None = None,
        parent: Adapter | None = None,
        context_key: str | None = None,
        description: str | None = None,
        config: dict[str, Any] | None = None,
        deployment_policy: Literal["skip", "add", "replace"] | None = None,
    ) -> Manifest:
        """Create a new training manifest with inline data.

        Args:
            adapter: Output adapter key (series name, e.g., "my-agent-sft").
            method: Training method ("dpo" or "sft").
            data: List of training records (inline data).
            model: Optional base model (can be specified at training time via --model).
            parent: Parent adapter for lineage (continue training from this).
            context_key: Agent context key for provenance.
            description: Human-readable description.
            config: Override training configuration (num_epochs, etc.).
            deployment_policy: Deployment policy after training:
                - "skip": Don't deploy the adapter.
                - "add": Deploy as new version, keep existing versions.
                - "replace": Deploy and remove existing versions (default).

        Returns:
            Manifest instance.
        """
        source = Source(context_key=context_key, description=description)
        lora_config, training_config, method_config = self._build_manifest_configs(
            method, model, config
        )

        # Build deployment config (default to "replace" if not specified)
        deployment = Deployment(policy=deployment_policy or "replace")

        manifest = Manifest(
            adapter=adapter,
            method=method,
            data=Data(format="inline", records=data),
            deployment=deployment,
            source=source,
            parent=parent,
            lora=lora_config,
            training=training_config,
            method_config=method_config,
        )

        self._lg.info("created manifest", extra={"adapter": adapter, "method": method})
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

    def submit(self, manifest: Manifest) -> None:
        """Submit a manifest to the pending training queue.

        Saves the manifest to the registry's pending directory.

        Args:
            manifest: Manifest to submit.

        Raises:
            ValueError: If manifest with same key already pending.
        """
        self._storage.submit_manifest(manifest)
        self._lg.info("submitted manifest", extra={"adapter": manifest.adapter})

    def list_pending(self) -> list[Manifest]:
        """List manifests waiting for training.

        Returns:
            List of pending manifests.
        """
        return self._storage.list_pending_manifests()

    def list_completed(self) -> list[Manifest]:
        """List completed manifests.

        Returns:
            List of completed manifests.
        """
        return self._storage.list_completed_manifests()

    def _validate_adapter_name(self, adapter: str) -> None:
        """Validate adapter name has no path traversal characters."""
        self._storage.validate_key(adapter)

    def get_pending(self, adapter: str) -> Manifest | None:
        """Get a pending manifest by adapter key.

        Args:
            adapter: Adapter key to look up.

        Returns:
            Manifest or None if not found.
        """
        return self._storage.get_pending_manifest(adapter)

    def remove_pending(self, adapter: str) -> None:
        """Remove a manifest from the pending queue.

        Args:
            adapter: Adapter key of manifest to remove.

        Raises:
            FileNotFoundError: If manifest not in queue.
            ValueError: If adapter name contains path traversal characters.
        """
        self._storage.remove_pending_manifest(adapter)
        self._lg.info("removed from queue", extra={"adapter": adapter})

    def find_adapter(self, md5: str) -> Adapter | None:
        """Find an adapter by its md5 hash.

        Searches completed manifests for an adapter with matching md5.

        Args:
            md5: MD5 hash of adapter weights (12 char hex).

        Returns:
            Adapter if found, None otherwise.
        """
        manifest = self._storage.find_adapter_by_md5(md5)
        if manifest and manifest.output and manifest.output.adapter:
            adapter: Adapter = manifest.output.adapter
            return adapter
        return None

    def get_latest_completed(
        self, adapter: str | None = None, context_key: str | None = None
    ) -> Manifest | None:
        """Get the most recently completed manifest.

        Args:
            adapter: Filter by adapter key.
            context_key: Filter by agent context key.

        Returns:
            Most recent completed manifest matching filters, or None.
        """
        manifests: list[Manifest] = []
        for manifest in self.list_completed():
            if adapter and manifest.adapter != adapter:
                continue
            if context_key and manifest.source.context_key != context_key:
                continue
            if manifest.output and manifest.output.status == "completed":
                manifests.append(manifest)

        if not manifests:
            return None

        # Sort by completed_at descending, return most recent
        return max(
            manifests,
            key=lambda m: (
                m.output.completed_at if m.output and m.output.completed_at else m.created_at
            ),
        )
