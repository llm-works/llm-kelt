"""DPO training client - executes DPO training from manifests.

Provides methods to train adapters using Direct Preference Optimization.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from appinfra import DotDict
from appinfra.log import Logger

from ..manifest.schema import Manifest, get_deploy_setting
from ..schema import RunResult
from ..storage import Storage

if TYPE_CHECKING:
    from ..lora.config import Config
    from ..lora.registry import AdapterRegistry


class Client:
    """Client for DPO (Direct Preference Optimization) training.

    Executes DPO training from manifest objects or paths.
    Accessed via Client.train.dpo.

    Usage:
        # Train from manifest object
        manifest = kelt.train.manifest.create(...)
        result = kelt.train.dpo.train(manifest)

        # Train from manifest path
        result = kelt.train.dpo.train_from_path(Path("manifest.yaml"))
    """

    def __init__(
        self,
        lg: Logger,
        storage: Storage,
    ) -> None:
        """Initialize DPO client.

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
            from ..lora.registry import AdapterRegistry

            self._registry = AdapterRegistry(self._lg, self._storage)
        return self._registry

    def _validate_parent(self, manifest: Manifest) -> None:
        """Validate parent adapter exists."""
        if manifest.parent is None:
            return

        parent_path = Path(manifest.parent.path)
        if not parent_path.exists():
            raise ValueError(f"Parent adapter not found: {manifest.parent.path}")

        self._lg.info(
            "using parent adapter",
            extra={"path": str(parent_path), "md5": manifest.parent.md5},
        )

    def _validate_records(self, manifest: Manifest) -> None:
        """Validate DPO record structure (inline data only; external files validated at load)."""
        errors: list[str] = []
        for i, record in enumerate(manifest.data.records):
            if "prompt" not in record:
                errors.append(f"DPO record {i} missing 'prompt' field")
            if "chosen" not in record:
                errors.append(f"DPO record {i} missing 'chosen' field")
            if "rejected" not in record:
                errors.append(f"DPO record {i} missing 'rejected' field")
        if errors:
            raise ValueError(f"Invalid DPO data: {'; '.join(errors)}")

    def _build_lora_config(self, lora: DotDict, base_model: str, training: DotDict) -> Config:
        """Build lora.Config from manifest DotDict with model-aware defaults."""
        from ..profiles import build_lora_config

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
        work_dir = self._storage.create_work_area(manifest.adapter, clean=True)
        data_path = self._storage.resolve_data_path(manifest, work_dir)
        self._lg.info("resolved data", extra={"path": str(data_path)})

        self._validate_parent(manifest)
        return work_dir, data_path

    def _log_training_start(self, manifest: Manifest, base_model: str) -> None:
        """Log DPO training start."""
        self._lg.info(
            "starting DPO training",
            extra={
                "adapter": manifest.adapter,
                "model": base_model,
                "epochs": manifest.training.get("num_epochs", 3),
                "beta": manifest.method_config.get("beta", 0.1),
                "parent": manifest.parent.path if manifest.parent else None,
            },
        )

    def _execute_dpo(
        self,
        manifest: Manifest,
        work_dir: Path,
        data_path: Path,
    ) -> RunResult:
        """Execute DPO training."""
        from .trainer import train_dpo

        base_model: str | None = manifest.training.get("base_model")
        if not base_model:
            raise ValueError(
                "Manifest missing 'base_model' in training config. "
                "Provide base_model when creating the manifest or set requested_model."
            )
        self._log_training_start(manifest, base_model)

        result = train_dpo(
            lg=self._lg,
            data_path=data_path,
            output_dir=work_dir,
            base_model=base_model,
            lora_config=self._build_lora_config(manifest.lora, base_model, manifest.training),
            training_config=manifest.training,
            beta=manifest.method_config.get("beta", 0.1),
            reference_free=manifest.method_config.get("reference_free", False),
            parent=manifest.parent,
        )

        self._lg.info(
            "training complete",
            extra={"duration_s": result.duration_seconds, "samples": result.samples_trained},
        )
        return result

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

        from ..schema import Adapter

        # Compute metadata before registration (caller receives result with populated md5/mtime)
        if result.adapter:
            meta = compute_adapter_metadata(Path(result.adapter.path))
            result.adapter = Adapter(md5=meta.md5, mtime=meta.mtime, path=result.adapter.path)

        description = (manifest.source.description if manifest.source else None) or "DPO adapter"
        deploy = get_deploy_setting(manifest)
        info = self.registry.register(
            training_result=result,
            key=manifest.adapter,
            description=description,
            deploy=deploy,
            overwrite=True,
        )
        # Update result.adapter.path to registered location (adapters/, not work/)
        if result.adapter:
            result.adapter = Adapter(
                md5=result.adapter.md5, mtime=result.adapter.mtime, path=info.path
            )
        self._lg.info("registered adapter", extra={"adapter": manifest.adapter})

    def train(self, manifest: Manifest, *, register: bool = True) -> RunResult:
        """Execute DPO training from a manifest.

        Args:
            manifest: Manifest with DPO configuration and data.
            register: If True, register adapter to registry after training.

        Returns:
            RunResult with training metrics and adapter path.

        Raises:
            ValueError: If manifest method is not 'dpo'.
        """
        if manifest.method != "dpo":
            raise ValueError(f"Expected DPO manifest, got method='{manifest.method}'")

        if manifest.data.format == "inline":
            self._validate_records(manifest)

        work_dir, data_path = self._prepare_training(manifest)
        result = self._execute_dpo(manifest, work_dir, data_path)

        if register:
            self._register_adapter(result, manifest)

        return result
