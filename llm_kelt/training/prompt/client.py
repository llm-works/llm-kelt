"""Prompt tuning client - executes prompt tuning training from manifests.

Provides methods to train adapters using Prompt Tuning (extremely parameter-efficient).
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
    from ..lora.registry import AdapterRegistry
    from .config import Config


class Client:
    """Client for Prompt Tuning training.

    Prompt tuning is extremely parameter-efficient (~50K params vs ~50M for LoRA).
    Ideal for large models (32B+) with small datasets where LoRA is unstable.

    Usage:
        manifest = kelt.train.manifest.create(method="prompt", ...)
        result = kelt.train.prompt.train(manifest)
    """

    def __init__(self, lg: Logger, storage: Storage) -> None:
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

    def _validate_records(self, manifest: Manifest) -> None:
        """Validate prompt tuning record structure."""
        errors: list[str] = []
        for i, record in enumerate(manifest.data.records):
            if "instruction" not in record and "prompt" not in record:
                errors.append(f"Record {i} missing 'instruction' or 'prompt' field")
            if "output" not in record and "response" not in record:
                errors.append(f"Record {i} missing 'output' or 'response' field")
        if errors:
            raise ValueError(f"Invalid data: {'; '.join(errors)}")

    def _build_prompt_config(self, prompt: DotDict) -> Config:
        """Build prompt.Config from manifest DotDict."""
        from .config import Config

        num_virtual_tokens = prompt.get("num_virtual_tokens", 20)
        if num_virtual_tokens > 100:
            self._lg.warning(
                "num_virtual_tokens > 100 is unusual and may hurt performance",
                extra={"num_virtual_tokens": num_virtual_tokens},
            )
        return Config(
            num_virtual_tokens=num_virtual_tokens,
            prompt_tuning_init=prompt.get("prompt_tuning_init", "TEXT"),
            prompt_tuning_init_text=prompt.get(
                "prompt_tuning_init_text", "You are a helpful assistant."
            ),
        )

    def _get_base_model(self, manifest: Manifest) -> str:
        """Get and validate base model from manifest."""
        base_model: str | None = manifest.training.get("base_model")
        if not base_model:
            base_model = manifest.training.get("requested_model")
        if not base_model:
            raise ValueError("Manifest missing 'base_model' in training config.")
        return base_model

    def _execute_training(
        self,
        manifest: Manifest,
        work_dir: Path,
        data_path: Path,
    ) -> RunResult:
        """Execute prompt tuning training."""
        from .trainer import Trainer

        base_model = self._get_base_model(manifest)
        prompt_config = self._build_prompt_config(manifest.method_config)

        self._lg.info(
            "starting prompt tuning",
            extra={
                "adapter": manifest.adapter,
                "model": base_model,
                "num_virtual_tokens": prompt_config.num_virtual_tokens,
            },
        )

        trainer = Trainer(
            lg=self._lg,
            data_path=data_path,
            output_dir=work_dir,
            base_model=base_model,
            prompt_config=prompt_config,
            training_config=manifest.training,
        )
        result = trainer.train()

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

        description = (manifest.source.description if manifest.source else None) or "Prompt adapter"
        deploy = get_deploy_setting(manifest)
        info = self.registry.register(
            training_result=result,
            key=manifest.adapter,
            description=description,
            deploy=deploy,
            overwrite=True,
        )
        if result.adapter and info.path != result.adapter.path:
            meta = compute_adapter_metadata(Path(info.path))
            result.adapter = Adapter(md5=meta.md5, mtime=meta.mtime, path=info.path)
        self._lg.info("registered adapter", extra={"adapter": manifest.adapter})

    def train(self, manifest: Manifest, *, register: bool = True) -> RunResult:
        """Execute prompt tuning training from a manifest.

        Args:
            manifest: Manifest with prompt tuning configuration and data.
            register: If True, register adapter to registry after training.

        Returns:
            RunResult with training metrics and adapter path.
        """
        if manifest.method != "prompt":
            raise ValueError(f"Expected prompt manifest, got method='{manifest.method}'")

        if manifest.data.format == "inline":
            self._validate_records(manifest)

        work_dir = self._storage.create_work_area(manifest.adapter, clean=True)
        data_path = self._storage.resolve_data_path(manifest, work_dir)
        self._lg.info("resolved data", extra={"path": str(data_path)})

        result = self._execute_training(manifest, work_dir, data_path)

        if register:
            self._register_adapter(result, manifest)

        return result
