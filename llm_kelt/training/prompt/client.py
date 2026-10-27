# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-kelt Authors

"""Prompt tuning client - executes prompt tuning training from manifests.

Provides methods to train adapters using Prompt Tuning (extremely parameter-efficient).
"""

from __future__ import annotations

from pathlib import Path

from appinfra import DotDict

from ..manifest.schema import Manifest
from ..method_client import MethodClient
from ..runtime import load
from ..schema import Adapter, RunResult
from .config import Config


class Client(MethodClient):
    """Client for Prompt Tuning training.

    Prompt tuning is extremely parameter-efficient (~50K params vs ~50M for LoRA).
    Ideal for large models (32B+) with small datasets where LoRA is unstable.

    Usage:
        manifest = kelt.train.manifest.create(method="prompt", ...)
        result = kelt.train.prompt.train(manifest)
    """

    METHOD = "prompt"
    LABEL = "Prompt"

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

    def _execute(self, manifest: Manifest, work_dir: Path, data_path: Path) -> RunResult:
        """Execute prompt tuning training."""
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

        trainer = load("prompt").Trainer(
            lg=self._lg,
            data_path=data_path,
            output_dir=work_dir,
            base_model=base_model,
            prompt_config=prompt_config,
            training_config=manifest.training,
        )
        result: RunResult = trainer.train()

        self._lg.info("training complete", extra={"duration_s": result.duration_seconds})
        return result

    def _registered_adapter(self, adapter: Adapter, path: str) -> Adapter:
        """The adapter as registered, md5/mtime read from the registered copy when it moved."""
        if path == adapter.path:
            return adapter
        from llm_infer import compute_adapter_metadata

        meta = compute_adapter_metadata(Path(path))
        return Adapter(md5=meta.md5, mtime=meta.mtime, path=path)
