"""Manifest client - lifecycle management for training manifests.

Handles creating, loading, saving, and submitting manifests to the training queue.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from appinfra import DotDict
from appinfra.log import Logger

from ..schema import Adapter
from .errors import CorruptedManifestError
from .loader import load_manifest as _load_manifest
from .loader import save_manifest as _save_manifest
from .schema import Data, Manifest, Source

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
        registry_path: Path,
        default_profiles: dict[str, dict] | None = None,
    ) -> None:
        """Initialize manifest client.

        Args:
            lg: Logger instance.
            registry_path: Base path for adapter registry and manifest queues.
            default_profiles: Default training profiles by method (e.g., {"sft": {...}, "dpo": {...}}).
        """
        self._lg = lg
        self._registry_path = Path(registry_path).expanduser()
        self._default_profiles = default_profiles or {}

    def _ensure_dirs(self) -> None:
        """Lazily create directory structure on first write operation."""
        for subdir in ("pending", "completed", "adapters", "deployed"):
            (self._registry_path / subdir).mkdir(parents=True, exist_ok=True)

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

        Returns:
            Manifest instance.
        """
        source = Source(context_key=context_key, description=description)
        lora_config, training_config, method_config = self._build_manifest_configs(
            method, model, config
        )

        manifest = Manifest(
            adapter=adapter,
            method=method,
            data=Data(format="inline", records=data),
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

    def submit(self, manifest: Manifest) -> Path:
        """Submit a manifest to the pending training queue.

        Saves the manifest to the registry's pending directory.

        Args:
            manifest: Manifest to submit.

        Returns:
            Path to manifest in pending queue.

        Raises:
            ValueError: If manifest with same key already pending.
        """
        self._ensure_dirs()

        # Validate adapter name to prevent path traversal
        adapter = manifest.adapter
        if "/" in adapter or "\\" in adapter or ".." in adapter:
            raise ValueError(f"Invalid adapter name (path traversal): {adapter}")

        pending_dir = self._registry_path / "pending"
        dest_path = pending_dir / f"{adapter}.yaml"

        # Note: check-then-write has TOCTOU race, but acceptable for single-user CLI
        if dest_path.exists():
            raise ValueError(f"Manifest already in queue: {adapter}")

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
        """List completed manifests (supports .yaml and .yaml.gz).

        Returns:
            List of paths to completed manifest files.
        """
        completed_dir = self._registry_path / "completed"
        yaml_files = list(completed_dir.glob("*.yaml"))
        gz_files = list(completed_dir.glob("*.yaml.gz"))
        return sorted(yaml_files + gz_files)

    def get_pending(self, adapter: str) -> Manifest | None:
        """Get a pending manifest by adapter key.

        Args:
            adapter: Adapter key to look up.

        Returns:
            Manifest or None if not found or corrupted.
        """
        path = self._registry_path / "pending" / f"{adapter}.yaml"
        if not path.exists():
            return None
        try:
            return _load_manifest(path)
        except CorruptedManifestError:
            self._lg.warning(
                "corrupted manifest, treating as non-existent", extra={"path": str(path)}
            )
            return None

    def remove_pending(self, adapter: str) -> None:
        """Remove a manifest from the pending queue.

        Args:
            adapter: Adapter key of manifest to remove.

        Raises:
            FileNotFoundError: If manifest not in queue.
        """
        pending_path = self._registry_path / "pending" / f"{adapter}.yaml"
        if not pending_path.exists():
            raise FileNotFoundError(f"Manifest not in queue: {adapter}")
        pending_path.unlink()
        self._lg.info("removed from queue", extra={"adapter": adapter})

    def find_adapter(self, md5: str) -> Adapter | None:
        """Find an adapter by its md5 hash.

        Searches completed manifests for an adapter with matching md5.

        Args:
            md5: MD5 hash of adapter weights (12 char hex).

        Returns:
            Adapter if found, None otherwise.
        """
        for path in self.list_completed():
            try:
                manifest = _load_manifest(path)
            except CorruptedManifestError:
                self._lg.warning("corrupted manifest, skipping", extra={"path": str(path)})
                continue
            if manifest.output and manifest.output.adapter and manifest.output.adapter.md5 == md5:
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
        for path in self.list_completed():
            try:
                manifest = _load_manifest(path)
            except CorruptedManifestError:
                self._lg.warning("corrupted manifest, skipping", extra={"path": str(path)})
                continue
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
