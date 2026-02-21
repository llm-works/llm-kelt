"""Manifest runner - execute training from manifest files.

Thin dispatcher that loads a manifest and delegates to the appropriate
training client (DPO or SFT) based on the manifest method.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from appinfra.log import Logger

from ..config import RunResult
from .loader import load_manifest, validate_manifest


@dataclass
class Result:
    """Result of manifest-based training.

    Attributes:
        training_result: Underlying RunResult from trainer.
        adapter_id: ID of the trained adapter.
        completed_path: Path where manifest was archived.
    """

    training_result: RunResult
    adapter_id: str
    completed_path: Path


class Runner:
    """Executes training from manifest files.

    Loads manifests, validates them, and dispatches to the appropriate
    training client (DPO or SFT) based on the manifest method.
    """

    def __init__(self, lg: Logger, registry_path: Path) -> None:
        """Initialize runner.

        Args:
            lg: Logger instance.
            registry_path: Base path for adapter registry.
        """
        self._lg = lg
        self._registry_path = Path(registry_path).expanduser()

    def _move_to_completed(self, manifest_path: Path) -> Path:
        """Move manifest from pending to completed directory."""
        completed_dir = self._registry_path / "completed"
        completed_dir.mkdir(parents=True, exist_ok=True)

        completed_path = completed_dir / manifest_path.name
        if completed_path.exists():
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            completed_path = (
                completed_dir / f"{manifest_path.stem}-{timestamp}{manifest_path.suffix}"
            )

        shutil.move(str(manifest_path), str(completed_path))
        self._lg.info("moved manifest to completed", extra={"path": str(completed_path)})
        return completed_path

    def run(
        self,
        manifest_path: Path,
        output_dir: Path | None = None,
        skip_registration: bool = False,
    ) -> Result:
        """Execute training from a manifest file.

        Args:
            manifest_path: Path to manifest YAML file.
            output_dir: Working directory for training output. Defaults to registry/work.
            skip_registration: If True, skip copying to registry (for testing).

        Returns:
            Result with training result, adapter ID, and completed path.
        """
        self._lg.info("loading manifest", extra={"path": str(manifest_path)})
        manifest = load_manifest(manifest_path)

        errors = validate_manifest(manifest)
        if errors:
            raise ValueError(f"Invalid manifest: {'; '.join(errors)}")

        # Dispatch to appropriate client
        if manifest.method == "dpo":
            from ..dpo import Client as DpoClient

            training_result = DpoClient(self._lg, self._registry_path).train(
                manifest, output_dir=output_dir, register=not skip_registration
            )
        else:
            from ..sft import Client as SftClient

            training_result = SftClient(self._lg, self._registry_path).train(
                manifest, output_dir=output_dir, register=not skip_registration
            )

        completed_path = self._move_to_completed(manifest_path)
        return Result(
            training_result=training_result,
            adapter_id=manifest.adapter_id,
            completed_path=completed_path,
        )
