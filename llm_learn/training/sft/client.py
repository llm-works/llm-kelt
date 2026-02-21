"""SFT training client - executes SFT training from manifests.

Provides methods to train adapters using Supervised Fine-Tuning.
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
    """Client for SFT (Supervised Fine-Tuning) training.

    Executes SFT training from manifest objects or paths.
    Accessed via LearnClient.train.sft.

    Usage:
        # Train from manifest object
        manifest = learn.train.manifest.create(method="sft", ...)
        result = learn.train.sft.train(manifest)

        # Train from manifest path
        result = learn.train.sft.train_from_path(Path("manifest.yaml"))
    """

    def __init__(
        self,
        lg: Logger,
        registry_path: Path,
    ) -> None:
        """Initialize SFT client.

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
        """Validate SFT record structure."""
        errors: list[str] = []
        for i, record in enumerate(manifest.data.records):
            if "instruction" not in record and "prompt" not in record:
                errors.append(f"SFT record {i} missing 'instruction' or 'prompt' field")
            if "output" not in record and "response" not in record:
                errors.append(f"SFT record {i} missing 'output' or 'response' field")
        if errors:
            raise ValueError(f"Invalid SFT data: {'; '.join(errors)}")

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

    def _execute_sft(
        self,
        manifest: Manifest,
        work_dir: Path,
        data_path: Path,
        parent_adapter: Path | None,
    ) -> RunResult:
        """Execute SFT training."""
        from ..lora.trainer import train_lora

        self._lg.info(
            "starting SFT training",
            extra={
                "adapter_id": manifest.adapter_id,
                "model": manifest.model.base,
                "epochs": manifest.training.num_epochs,
                "parent": str(parent_adapter) if parent_adapter else None,
            },
        )

        result = train_lora(
            lg=self._lg,
            data_path=data_path,
            output_dir=work_dir,
            base_model=manifest.model.base,
            lora_config=self._build_lora_config(manifest.lora),
            training_config=manifest.training,
            quantize=manifest.model.quantize,
            resume_from=parent_adapter,
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
        """Execute SFT training from a manifest.

        Args:
            manifest: Manifest with SFT configuration and data.
            output_dir: Working directory for training. Defaults to registry/work.
            register: If True, register adapter to registry after training.

        Returns:
            RunResult with training metrics and adapter path.

        Raises:
            ValueError: If manifest method is not 'sft'.
        """
        if manifest.method != "sft":
            raise ValueError(f"Expected SFT manifest, got method='{manifest.method}'")

        if manifest.data.format == "inline":
            self._validate_records(manifest)

        work_dir, data_path, parent_adapter = self._prepare_training(manifest, output_dir)
        result = self._execute_sft(manifest, work_dir, data_path, parent_adapter)

        if register:
            description = manifest.source.description or "SFT adapter"
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
        """Execute SFT training from a manifest file.

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
