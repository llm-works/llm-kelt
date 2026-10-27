# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-kelt Authors

"""SFT training client - executes SFT training from manifests.

Provides methods to train adapters using Supervised Fine-Tuning.
"""

from __future__ import annotations

import json
from pathlib import Path

from ...core.export.jsonl import load_jsonl
from ..manifest.schema import Manifest
from ..manifest.templates import render_records, validate_templates
from ..method_client import MethodClient
from ..runtime import load
from ..schema import RunResult


class Client(MethodClient):
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

    METHOD = "sft"
    LABEL = "SFT"

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
            if not any(k in record for k in ("instruction", "prompt", "template")):
                errors.append(f"SFT record {i} missing 'instruction', 'prompt' or 'template' field")
            if "output" not in record and "response" not in record:
                errors.append(f"SFT record {i} missing 'output' or 'response' field")
        if errors:
            raise ValueError(f"Invalid SFT data: {'; '.join(errors)}")

    def _resolve_data(self, manifest: Manifest, work_dir: Path, data_path: Path) -> Path:
        """Render templated records into a plain JSONL file; return data_path if none."""
        templates = manifest.data.templates
        if not templates:
            return data_path

        errors = validate_templates(templates)
        if errors:
            raise ValueError(f"Invalid SFT templates: {'; '.join(errors)}")

        records = render_records(templates, load_jsonl(data_path))
        rendered_path = work_dir / f"{manifest.adapter}-rendered.jsonl"
        with rendered_path.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        self._lg.info(
            "rendered templated records",
            extra={"templates": len(templates), "records": len(records)},
        )
        return rendered_path

    def _execute(self, manifest: Manifest, work_dir: Path, data_path: Path) -> RunResult:
        """Execute SFT training."""
        base_model = self._get_base_model(manifest)
        self._lg.info(
            "starting SFT training",
            extra={"adapter": manifest.adapter, "model": base_model},
        )

        result: RunResult = load("lora").train_lora(
            lg=self._lg,
            data_path=data_path,
            output_dir=work_dir,
            base_model=base_model,
            lora_config=self._build_lora_config(manifest.lora, base_model, manifest.training),
            training_config=manifest.training,
        )

        self._lg.info("training complete", extra={"duration_s": result.duration_seconds})
        return result
