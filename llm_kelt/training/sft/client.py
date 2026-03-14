"""SFT training client - executes SFT training from manifests.

Provides methods to train adapters using Supervised Fine-Tuning.
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
    """Client for SFT (Supervised Fine-Tuning) training.

    Executes SFT training from manifest objects or paths.
    Accessed via Client.train.sft.

    Usage:
        # Train from manifest object
        manifest = kelt.train.manifest.create(method="sft", ...)
        result = kelt.train.sft.train(manifest)

        # Train from manifest path
        result = kelt.train.sft.train_from_path(Path("manifest.yaml"))
    """

    def __init__(
        self,
        lg: Logger,
        storage: Storage,
    ) -> None:
        """Initialize SFT client.

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
        """Validate parent adapter if specified (not yet used in SFT)."""
        if manifest.parent is None:
            return

        # SFT doesn't currently support training on top of existing adapters
        # (unlike DPO). Log this for awareness.
        self._lg.warning(
            "parent adapter specified but SFT doesn't support adapter chaining yet",
            extra={"path": manifest.parent.path, "md5": manifest.parent.md5},
        )

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

    def _get_base_model(self, manifest: Manifest) -> str:
        """Get and validate base model from manifest."""
        base_model: str | None = manifest.training.get("base_model")
        if not base_model:
            raise ValueError(
                "Manifest missing 'base_model' in training config. "
                "Provide base_model when creating the manifest or set requested_model."
            )
        return base_model

    def _execute_sft(
        self,
        manifest: Manifest,
        work_dir: Path,
        data_path: Path,
    ) -> RunResult:
        """Execute SFT training."""
        from ..lora.trainer import train_lora

        base_model = self._get_base_model(manifest)
        self._lg.info(
            "starting SFT training",
            extra={"adapter": manifest.adapter, "model": base_model},
        )

        result = train_lora(
            lg=self._lg,
            data_path=data_path,
            output_dir=work_dir,
            base_model=base_model,
            lora_config=self._build_lora_config(manifest.lora, base_model, manifest.training),
            training_config=manifest.training,
        )

        self._lg.info("training complete", extra={"duration_s": result.duration_seconds})
        return result

    def _register_adapter(self, result: RunResult, manifest: Manifest) -> None:
        """Register trained adapter to registry."""
        try:
            from llm_infer import compute_adapter_metadata
        except ImportError as e:
            raise ImportError(
                "Adapter registration requires llm-infer package. "
                "Install with: pip install llm-infer, or use register=False."
            ) from e

        from ..schema import Adapter

        if result.adapter:
            meta = compute_adapter_metadata(Path(result.adapter.path))
            result.adapter = Adapter(md5=meta.md5, mtime=meta.mtime, path=result.adapter.path)

        description = (manifest.source.description if manifest.source else None) or "SFT adapter"
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
        """Execute SFT training from a manifest.

        Args:
            manifest: Manifest with SFT configuration and data.
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

        work_dir, data_path = self._prepare_training(manifest)
        result = self._execute_sft(manifest, work_dir, data_path)

        if register:
            self._register_adapter(result, manifest)

        return result
