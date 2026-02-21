"""Manifest runner - execute training from manifest files.

Thin dispatcher that loads a manifest and delegates to the appropriate
training client (DPO or SFT) based on the manifest method.
"""

from __future__ import annotations

from pathlib import Path

from appinfra.log import Logger
from llm_infer import compute_adapter_metadata

from .manifest.loader import load_manifest, save_manifest, validate_manifest
from .manifest.schema import Manifest
from .schema import Adapter, RunResult


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

    def _enrich_adapter(self, result: RunResult) -> RunResult:
        """Populate adapter md5/mtime from weights file."""
        if result.adapter is None:
            return result

        meta = compute_adapter_metadata(Path(result.adapter.path))
        result.adapter = Adapter(
            md5=meta.md5,
            mtime=meta.mtime,
            path=result.adapter.path,
        )
        return result

    def _save_completed(self, manifest: Manifest, manifest_path: Path) -> Path:
        """Save manifest with output to completed directory."""
        completed_dir = self._registry_path / "completed"
        completed_dir.mkdir(parents=True, exist_ok=True)

        md5 = (
            manifest.output.adapter.md5
            if manifest.output and manifest.output.adapter
            else "unknown"
        )
        completed_path = completed_dir / f"{manifest.adapter}-{md5}.yaml"

        save_manifest(manifest, completed_path)
        self._lg.info("saved completed manifest", extra={"path": str(completed_path)})

        if manifest_path.exists():
            manifest_path.unlink()

        return completed_path

    def _dispatch_training(
        self, manifest: Manifest, output_dir: Path | None, skip_registration: bool
    ) -> RunResult:
        """Dispatch to appropriate training client based on method."""
        if manifest.method == "dpo":
            from .dpo import Client as DpoClient

            return DpoClient(self._lg, self._registry_path).train(
                manifest, output_dir=output_dir, register=not skip_registration
            )

        from .sft import Client as SftClient

        return SftClient(self._lg, self._registry_path).train(
            manifest, output_dir=output_dir, register=not skip_registration
        )

    def _create_failed_result(self, error: str) -> RunResult:
        """Create a failed RunResult for error cases."""
        from datetime import datetime

        now = datetime.now().astimezone()
        return RunResult(
            status="failed",
            base_model="",
            method="",
            metrics={},
            config={},
            started_at=now,
            completed_at=now,
            samples_trained=0,
            error=error,
        )

    def run(
        self,
        manifest_path: Path,
        output_dir: Path | None = None,
        skip_registration: bool = False,
    ) -> RunResult:
        """Execute training from a manifest file.

        Args:
            manifest_path: Path to manifest YAML file.
            output_dir: Working directory for training.
            skip_registration: If True, don't register adapter to registry.

        Returns:
            RunResult with adapter metadata, metrics, and training info.
        """
        self._lg.info("loading manifest", extra={"path": str(manifest_path)})
        manifest = load_manifest(manifest_path)

        errors = validate_manifest(manifest)
        if errors:
            raise ValueError(f"Invalid manifest: {'; '.join(errors)}")

        try:
            result = self._dispatch_training(manifest, output_dir, skip_registration)
            result = self._enrich_adapter(result)
            manifest.output = result
        except Exception as e:
            manifest.output = self._create_failed_result(str(e))
            self._save_completed(manifest, manifest_path)
            raise

        self._save_completed(manifest, manifest_path)
        return result
