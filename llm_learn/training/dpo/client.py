"""DPO training client - executes DPO training from manifests.

Provides methods to train adapters using Direct Preference Optimization.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from appinfra import DotDict
from appinfra.log import Logger

from ..manifest.loader import resolve_data
from ..manifest.schema import Manifest
from ..schema import RunResult

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

    def _prepare_training(self, manifest: Manifest, output_dir: Path | None) -> tuple[Path, Path]:
        """Prepare training: resolve work dir and data path.

        Cleans any existing work directory to ensure fresh training state.
        """
        import shutil

        work_dir = output_dir or (self._registry_path / "work" / manifest.adapter)
        if work_dir.exists():
            self._lg.info("cleaning previous work dir", extra={"path": str(work_dir)})
            shutil.rmtree(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)

        data_path = resolve_data(manifest, work_dir)
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

        base_model = manifest.training["base_model"]
        self._log_training_start(manifest, base_model)

        result = train_dpo(
            lg=self._lg,
            data_path=data_path,
            output_dir=work_dir,
            base_model=base_model,
            lora_config=self._build_lora_config(manifest.lora),
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
        from llm_infer import compute_adapter_metadata

        from ..schema import Adapter

        # Intentionally replace result.adapter in-place with computed metadata
        # before registration. Caller receives result with populated md5/mtime.
        if result.adapter:
            meta = compute_adapter_metadata(Path(result.adapter.path))
            result.adapter = Adapter(md5=meta.md5, mtime=meta.mtime, path=result.adapter.path)

        description = manifest.source.description or "DPO adapter"
        self.registry.register(
            training_result=result,
            key=manifest.adapter,
            description=description,
            deploy=True,
            overwrite=True,
        )
        self._lg.info("registered adapter", extra={"adapter": manifest.adapter})

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

        work_dir, data_path = self._prepare_training(manifest, output_dir)
        result = self._execute_dpo(manifest, work_dir, data_path)

        if register:
            self._register_adapter(result, manifest)

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
