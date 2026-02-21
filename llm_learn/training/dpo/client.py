"""DPO training client - executes DPO training from manifests.

Provides methods to train adapters using Direct Preference Optimization.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from appinfra import DotDict
from appinfra.log import Logger

from ..config import RunResult
from ..manifest.loader import resolve_data
from ..manifest.schema import Manifest

if TYPE_CHECKING:
    from ..lora.config import Config
    from ..lora.registry import AdapterRegistry


class Client:
    """Client for DPO (Direct Preference Optimization) training.

    Executes DPO training from manifest objects or paths.
    Accessed via LearnClient.train.dpo.

    Usage:
        # Train from manifest object
        manifest = learn.train.manifest.create(...)
        result = learn.train.dpo.train(manifest)

        # Train from manifest path
        result = learn.train.dpo.train_from_path(Path("manifest.yaml"))
    """

    def __init__(
        self,
        lg: Logger,
        registry_path: Path,
    ) -> None:
        """Initialize DPO client.

        Args:
            lg: Logger instance.
            registry_path: Base path for adapter registry.
        """
        self._lg = lg
        self._registry_path = Path(registry_path).expanduser()
        self._registry: AdapterRegistry | None = None

    @property
    def registry(self) -> AdapterRegistry:
        """Get or create AdapterRegistry instance."""
        if self._registry is None:
            from ..lora.registry import AdapterRegistry

            self._registry = AdapterRegistry(self._lg, self._registry_path)
        return self._registry

    def _resolve_parent_adapter(self, manifest: Manifest) -> Path | None:
        """Resolve parent adapter reference to path."""
        if manifest.parent_adapter is None:
            return None

        parent_ref = manifest.parent_adapter

        # Try as direct path first
        parent_path = Path(parent_ref)
        if parent_path.exists():
            self._lg.info("resolved parent adapter from path", extra={"path": str(parent_path)})
            return parent_path

        # Try registry lookup
        try:
            resolved = self.registry.resolve(parent_ref)
            self._lg.info("resolved parent adapter from registry", extra={"id": parent_ref})
            return resolved
        except ValueError:
            pass

        raise ValueError(f"Parent adapter not found: {parent_ref}")

    def _validate_records(self, manifest: Manifest) -> None:
        """Validate DPO record structure."""
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

    def _build_lora_config(self, lora: DotDict) -> Config:
        """Build lora.Config from manifest DotDict."""
        from ..lora.config import Config

        defaults = Config()
        return Config(
            r=lora.get("r", defaults.r),
            lora_alpha=lora.get("lora_alpha", defaults.lora_alpha),
            lora_dropout=lora.get("lora_dropout", defaults.lora_dropout),
            target_modules=lora.get("target_modules", defaults.target_modules),
            bias=lora.get("bias", defaults.bias),
            task_type=lora.get("task_type", defaults.task_type),
        )

    def _prepare_training(
        self, manifest: Manifest, output_dir: Path | None
    ) -> tuple[Path, Path, Path | None]:
        """Prepare training: resolve work dir, data path, and parent adapter."""
        work_dir = output_dir or (self._registry_path / "work" / manifest.adapter_id)
        work_dir.mkdir(parents=True, exist_ok=True)

        data_path = resolve_data(manifest, work_dir)
        self._lg.info("resolved data", extra={"path": str(data_path)})

        parent_adapter = self._resolve_parent_adapter(manifest)
        return work_dir, data_path, parent_adapter

    def _log_training_start(self, manifest: Manifest, parent: Path | None) -> None:
        """Log DPO training start."""
        self._lg.info(
            "starting DPO training",
            extra={
                "adapter_id": manifest.adapter_id,
                "model": manifest.model.base,
                "epochs": manifest.training.num_epochs,
                "beta": manifest.method_config.get("beta", 0.1),
                "parent": str(parent) if parent else None,
            },
        )

    def _execute_dpo(
        self,
        manifest: Manifest,
        work_dir: Path,
        data_path: Path,
        parent_adapter: Path | None,
    ) -> RunResult:
        """Execute DPO training."""
        from .trainer import train_dpo

        self._log_training_start(manifest, parent_adapter)

        result = train_dpo(
            lg=self._lg,
            data_path=data_path,
            output_dir=work_dir,
            base_model=manifest.model.base,
            lora_config=self._build_lora_config(manifest.lora),
            training_config=manifest.training,
            beta=manifest.method_config.get("beta", 0.1),
            quantize=manifest.model.quantize,
            reference_free=manifest.method_config.get("reference_free", False),
            based_on=parent_adapter,
        )

        self._lg.info(
            "training complete",
            extra={"duration_s": result.duration_seconds, "samples": result.samples_trained},
        )
        return result

    def train(
        self,
        manifest: Manifest,
        *,
        output_dir: Path | None = None,
        register: bool = True,
    ) -> RunResult:
        """Execute DPO training from a manifest.

        Args:
            manifest: Manifest with DPO configuration and data.
            output_dir: Working directory for training. Defaults to registry/work.
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

        work_dir, data_path, parent_adapter = self._prepare_training(manifest, output_dir)
        result = self._execute_dpo(manifest, work_dir, data_path, parent_adapter)

        if register:
            description = manifest.source.description or "DPO adapter"
            self.registry.register(
                training_result=result,
                adapter_id=manifest.adapter_id,
                description=description,
                deploy=True,
                overwrite=True,
            )
            self._lg.info("registered adapter", extra={"adapter_id": manifest.adapter_id})

        return result

    def train_from_path(
        self,
        manifest_path: Path,
        *,
        output_dir: Path | None = None,
        register: bool = True,
    ) -> RunResult:
        """Execute DPO training from a manifest file.

        Args:
            manifest_path: Path to manifest YAML file.
            output_dir: Working directory for training.
            register: If True, register adapter after training.

        Returns:
            RunResult with training metrics and adapter path.
        """
        from ..manifest.loader import load_manifest

        manifest = load_manifest(manifest_path)
        return self.train(manifest, output_dir=output_dir, register=register)
