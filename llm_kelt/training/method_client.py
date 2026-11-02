# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-kelt Authors

"""Base of the per-method training clients: run a manifest's training, register the adapter."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from appinfra import DotDict
from appinfra.log import Logger

from .manifest.schema import Manifest, get_deploy_setting
from .schema import Adapter, RunResult
from .storage import Storage, queue_key

if TYPE_CHECKING:
    from .lora.config import Config as LoraConfig
    from .lora.registry import AdapterRegistry


class MethodClient:
    """Trains a manifest of one training method and registers the resulting adapter.

    Subclasses validate the method's records and run its trainer; they may also transform
    the resolved data file and check the manifest's parent adapter.
    """

    METHOD: str  # manifest.method this client trains
    LABEL: str  # names the method in messages and the default adapter description

    def __init__(self, lg: Logger, storage: Storage) -> None:
        """Initialize the client.

        Args:
            lg: Logger instance.
            storage: Storage instance for filesystem operations.
        """
        self._lg = lg
        self._storage = storage
        self._registry: AdapterRegistry | None = None

    @property
    def registry(self) -> AdapterRegistry:
        """Get or create AdapterRegistry instance."""
        if self._registry is None:
            from .lora.registry import AdapterRegistry

            self._registry = AdapterRegistry(self._lg, self._storage)
        return self._registry

    def _validate_records(self, manifest: Manifest) -> None:
        """Validate inline record structure (external files are validated at load)."""
        raise NotImplementedError

    def _execute(self, manifest: Manifest, work_dir: Path, data_path: Path) -> RunResult:
        """Run the method's training on the resolved data."""
        raise NotImplementedError

    def _resolve_data(self, manifest: Manifest, work_dir: Path, data_path: Path) -> Path:
        """Data file the trainer reads; the resolved file unless the method transforms it."""
        return data_path

    def _validate_parent(self, manifest: Manifest) -> None:
        """Check the manifest's parent adapter; methods that build on one override this."""

    def _get_base_model(self, manifest: Manifest) -> str:
        """Get and validate base model from manifest."""
        base_model: str | None = manifest.training.get("base_model")
        if not base_model:
            raise ValueError(
                "Manifest missing 'base_model' in training config. "
                "Provide base_model when creating the manifest or set requested_model."
            )
        return base_model

    def _build_lora_config(self, lora: DotDict, base_model: str, training: DotDict) -> LoraConfig:
        """Build lora.Config from manifest DotDict with model-aware defaults."""
        from .profiles import build_lora_config

        profile_name, config = build_lora_config(lora, base_model, training)
        self._lg.info(
            f"lora config (profile={profile_name})",
            extra={
                "r": config.r,
                "lora_alpha": config.lora_alpha,
                "lora_dropout": config.lora_dropout,
            },
        )
        return config

    def _prepare_training(self, manifest: Manifest) -> tuple[Path, Path]:
        """Prepare training: resolve work dir and data path."""
        work_dir = self._storage.create_work_area(
            queue_key(manifest.adapter, manifest.series), clean=True
        )
        data_path = self._storage.resolve_data_path(manifest, work_dir)
        data_path = self._resolve_data(manifest, work_dir, data_path)
        self._lg.info("resolved data", extra={"path": str(data_path)})

        self._validate_parent(manifest)
        return work_dir, data_path

    def _registered_adapter(self, adapter: Adapter, path: str) -> Adapter:
        """The trained adapter as registered at ``path``."""
        return Adapter(md5=adapter.md5, mtime=adapter.mtime, path=path)

    def _register_adapter(self, result: RunResult, manifest: Manifest) -> None:
        """Register trained adapter to registry.

        Computes md5 before registration (needed for versioned paths).
        Note: Runner also computes this, but Client can be used standalone.
        """
        try:
            from llm_infer import compute_adapter_metadata
        except ImportError as e:
            raise ImportError(
                "Adapter registration requires llm-infer package. "
                "Install with: pip install llm-infer, or use register=False."
            ) from e

        # Compute metadata before registration (caller receives result with populated md5/mtime)
        if result.adapter:
            meta = compute_adapter_metadata(Path(result.adapter.path))
            result.adapter = Adapter(md5=meta.md5, mtime=meta.mtime, path=result.adapter.path)

        source_description = manifest.source.description if manifest.source else None
        result.series, result.commit = manifest.series, manifest.commit
        info = self.registry.register(
            training_result=result,
            key=manifest.adapter,
            description=source_description or f"{self.LABEL} adapter",
            deploy=get_deploy_setting(manifest),
            overwrite=True,
        )
        # Point result.adapter at the registered location (adapters/, not work/)
        if result.adapter:
            result.adapter = self._registered_adapter(result.adapter, info.path)
        self._lg.info("registered adapter", extra={"adapter": manifest.adapter})

    def train(self, manifest: Manifest, *, register: bool = True) -> RunResult:
        """Execute training from a manifest.

        Args:
            manifest: Manifest with this method's configuration and data.
            register: If True, register adapter to registry after training.

        Returns:
            RunResult with training metrics and adapter path.

        Raises:
            ValueError: If the manifest's method is not this client's.
        """
        if manifest.method != self.METHOD:
            raise ValueError(f"Expected {self.LABEL} manifest, got method='{manifest.method}'")

        if manifest.data.format == "inline":
            self._validate_records(manifest)

        work_dir, data_path = self._prepare_training(manifest)
        result = self._execute(manifest, work_dir, data_path)

        if register:
            self._register_adapter(result, manifest)

        return result
