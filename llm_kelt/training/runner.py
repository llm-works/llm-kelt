"""Manifest runner - execute training from manifest files.

Thin dispatcher that loads a manifest and delegates to the appropriate
training client (DPO or SFT) based on the manifest method.
"""

from __future__ import annotations

from pathlib import Path

from appinfra.log import Logger
from llm_infer import compute_adapter_metadata
from llm_infer.models import ModelResolver

from .manifest.loader import load_manifest, validate_manifest
from .manifest.schema import Manifest
from .schema import Adapter, RunResult
from .storage import Storage


class Runner:
    """Executes training from manifest files.

    Loads manifests, validates them, and dispatches to the appropriate
    training client (DPO or SFT) based on the manifest method.
    """

    def __init__(
        self,
        lg: Logger,
        storage: Storage,
        model_locations: list[Path] | None = None,
    ) -> None:
        """Initialize runner.

        Args:
            lg: Logger instance.
            storage: Storage instance for filesystem operations.
            model_locations: Directories to search for models (for name resolution).
        """
        self._lg = lg
        self._storage = storage
        self._model_locations = model_locations or []

    def _resolve_model(self, model_name: str) -> str:
        """Resolve model name to full path if model_locations configured.

        If model_name is already an absolute path or HF ID (contains '/'),
        returns it unchanged. Otherwise searches model_locations.

        Args:
            model_name: Model name, path, or HF ID.

        Returns:
            Resolved model path or original name.
        """
        # Skip resolution if already a path or HF ID
        if "/" in model_name or Path(model_name).is_absolute():
            return model_name

        if not self._model_locations:
            self._lg.debug("no model_locations configured, using model name as-is")
            return model_name

        resolver = ModelResolver(lg=self._lg, locations=self._model_locations)
        resolved_path = resolver.find_by_name(model_name)

        if resolved_path is None:
            raise ValueError(f"Model not found: {model_name}")

        self._lg.info("resolved model", extra={"model": model_name, "path": str(resolved_path)})
        return str(resolved_path)

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

    def _save_completed(self, manifest: Manifest) -> None:
        """Save manifest with output to completed storage."""
        self._storage.complete_manifest(manifest)
        self._lg.info("saved completed manifest", extra={"adapter": manifest.adapter})

    def _dispatch_training(self, manifest: Manifest, skip_registration: bool) -> RunResult:
        """Dispatch to appropriate training client based on method."""
        if manifest.method == "dpo":
            from .dpo import Client as DpoClient

            return DpoClient(self._lg, self._storage).train(
                manifest, register=not skip_registration
            )

        # Fall through to SFT (method already validated by validate_manifest)
        from .sft import Client as SftClient

        return SftClient(self._lg, self._storage).train(manifest, register=not skip_registration)

    def _get_effective_model(self, manifest: Manifest, model_override: str | None) -> str:
        """Determine effective model from override or manifest.

        Args:
            manifest: Training manifest.
            model_override: CLI model override (takes precedence).

        Returns:
            Resolved model path.

        Raises:
            ValueError: If no model specified and none in manifest.
        """
        requested = manifest.training.get("requested_model")

        if model_override:
            self._lg.info("using model override", extra={"model": model_override})
            return self._resolve_model(model_override)

        if requested:
            self._lg.info("using requested model from manifest", extra={"model": requested})
            return self._resolve_model(requested)

        raise ValueError("No model specified. Use --model to specify the base model.")

    def run(
        self,
        manifest_path: Path,
        skip_registration: bool = False,
        model_override: str | None = None,
    ) -> RunResult:
        """Execute training from a manifest file.

        Args:
            manifest_path: Path to manifest YAML file.
            skip_registration: If True, don't register adapter to registry.
            model_override: Override manifest model (path, HF ID, or name to resolve).

        Returns:
            RunResult with adapter metadata, metrics, and training info.
        """
        self._lg.info("loading manifest", extra={"path": str(manifest_path)})
        manifest = load_manifest(manifest_path)

        errors = validate_manifest(manifest)
        if errors:
            raise ValueError(f"Invalid manifest: {'; '.join(errors)}")

        # Resolve effective model and store in training config
        base_model = self._get_effective_model(manifest, model_override)
        manifest.training["base_model"] = base_model

        result = self._dispatch_training(manifest, skip_registration)
        result = self._enrich_adapter(result)
        manifest.output = result

        self._save_completed(manifest)
        return result
