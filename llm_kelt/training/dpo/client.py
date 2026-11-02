# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-kelt Authors

"""DPO training client - executes DPO training from manifests.

Provides methods to train adapters using Direct Preference Optimization.
"""

from __future__ import annotations

from pathlib import Path

from ..manifest.schema import Manifest
from ..method_client import MethodClient
from ..runtime import load
from ..schema import RunResult


class Client(MethodClient):
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

    METHOD = "dpo"
    LABEL = "DPO"

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

    def _execute(self, manifest: Manifest, work_dir: Path, data_path: Path) -> RunResult:
        """Execute DPO training."""
        base_model = self._get_base_model(manifest)
        self._log_training_start(manifest, base_model)

        result: RunResult = load("dpo").train_dpo(
            lg=self._lg,
            data_path=data_path,
            output_dir=work_dir,
            base_model=base_model,
            lora_config=self._build_lora_config(manifest.lora, base_model, manifest.training),
            training_config=manifest.training,
            beta=manifest.method_config.get("beta", 0.1),
            parent=manifest.parent,
        )

        self._lg.info(
            "training complete",
            extra={"duration_s": result.duration_seconds, "samples": result.samples_trained},
        )
        return result
