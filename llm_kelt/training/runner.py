# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-kelt Authors

"""Manifest runner - execute training from manifest files.

Thin dispatcher that loads a manifest and delegates to the appropriate
training client (DPO, SFT, or Prompt Tuning) based on the manifest method.
"""

from __future__ import annotations

from datetime import UTC, datetime
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

        if manifest.method == "prompt":
            from .prompt import Client as PromptClient

            return PromptClient(self._lg, self._storage).train(
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

    def _prepare(
        self, manifest: Manifest, model_override: str | None, lora_profile: str | None
    ) -> None:
        """Validate the manifest and store the resolved base model and LoRA profile in it."""
        errors = validate_manifest(manifest)
        if errors:
            raise ValueError(f"Invalid manifest: {'; '.join(errors)}")

        manifest.training["base_model"] = self._get_effective_model(manifest, model_override)
        if lora_profile is not None:
            from .profiles import MODEL_SIZE_PROFILES

            if lora_profile not in MODEL_SIZE_PROFILES:
                valid = ", ".join(sorted(MODEL_SIZE_PROFILES.keys()))
                raise ValueError(f"Invalid lora_profile '{lora_profile}'. Valid values: {valid}")
            manifest.training["lora_profile"] = lora_profile

    def _train(
        self,
        manifest: Manifest,
        skip_registration: bool,
        model_override: str | None,
        lora_profile: str | None,
    ) -> RunResult:
        """Train a loaded manifest and save it to completed storage."""
        self._prepare(manifest, model_override, lora_profile)
        result = self._dispatch_training(manifest, skip_registration)
        result = self._enrich_adapter(result)
        manifest.output = result

        self._save_completed(manifest)
        return result

    def run(
        self,
        manifest_path: Path,
        skip_registration: bool = False,
        model_override: str | None = None,
        lora_profile: str | None = None,
    ) -> RunResult:
        """Execute training from a manifest file.

        Args:
            manifest_path: Path to manifest YAML file.
            skip_registration: If True, don't register adapter to registry.
            model_override: Override manifest model (path, HF ID, or name to resolve).
            lora_profile: LoRA profile override ("small", "medium", "large", "xlarge").
                If None, auto-detects from model size (aborts if detection fails).

        Returns:
            RunResult with adapter metadata, metrics, and training info.
        """
        self._lg.info("loading manifest", extra={"path": str(manifest_path)})
        manifest = load_manifest(manifest_path)
        return self._train(manifest, skip_registration, model_override, lora_profile)

    def run_pending(
        self,
        skip_registration: bool = False,
        model_override: str | None = None,
        lora_profile: str | None = None,
    ) -> list[tuple[str, RunResult]]:
        """Train every pending manifest, oldest submission first, one at a time.

        Each manifest is claimed (moved to running), trained, then moved to completed, or
        to failed with the error in its output. A failure does not stop the remaining
        manifests. Arguments are as for run().

        Returns:
            (adapter key, RunResult) per manifest trained or failed, in run order.
        """
        results: list[tuple[str, RunResult]] = []
        for adapter in self._storage.list_pending_keys():
            result = self._run_claimed(adapter, skip_registration, model_override, lora_profile)
            if result is not None:
                results.append((adapter, result))
        return results

    def _run_claimed(
        self,
        adapter: str,
        skip_registration: bool,
        model_override: str | None,
        lora_profile: str | None,
    ) -> RunResult | None:
        """Claim and train one pending manifest; None if it could not be claimed."""
        started_at = datetime.now(UTC)
        try:
            manifest = self._storage.claim_pending_manifest(adapter)
        except FileNotFoundError:
            self._lg.info("manifest no longer pending, skipping", extra={"adapter": adapter})
            return None
        except FileExistsError as e:
            self._lg.warning("manifest already running, skipping", extra={"exception": e})
            return None
        except Exception as e:  # claimed, but unreadable
            return self._fail(adapter, started_at, e)

        try:
            return self._train(manifest, skip_registration, model_override, lora_profile)
        except Exception as e:  # any training error fails this manifest, not the batch
            return self._fail(adapter, started_at, e)

    def _fail(self, adapter: str, started_at: datetime, error: Exception) -> RunResult:
        """Record a failed run and move the running manifest to failed storage."""
        result = RunResult(
            status="failed",
            started_at=started_at,
            completed_at=datetime.now(UTC),
            error=f"{type(error).__name__}: {error}",
        )
        location = self._storage.fail_running_manifest(adapter, result)
        self._lg.error(
            "training failed",
            extra={"adapter": adapter, "exception": error, "location": str(location)},
        )
        return result
