"""Adapter registry for llm-infer integration.

Thin wrapper around Storage that provides llm-infer API integration.
All adapter storage operations are delegated to the Storage implementation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import httpx
from appinfra.log import Logger

from ..schema import AdapterInfo
from ..storage import Storage

if TYPE_CHECKING:
    from ..schema import RunResult


class AdapterRegistry:
    """Registry for managing LoRA adapters with llm-infer.

    Thin wrapper around Storage that adds llm-infer API integration.
    All storage operations are delegated to the Storage instance.
    """

    def __init__(
        self,
        lg: Logger,
        storage: Storage,
        infer_url: str = "http://localhost:8000",
    ):
        """Initialize adapter registry.

        Args:
            lg: Logger instance.
            storage: Storage instance for all operations.
            infer_url: Base URL for llm-infer API.
        """
        self._lg = lg
        self._storage = storage
        self.infer_url = infer_url.rstrip("/")

    def register(
        self,
        training_result: RunResult,
        key: str,
        description: str | None = None,
        deploy: bool = True,
        overwrite: bool = False,
    ) -> AdapterInfo:
        """Register a trained adapter.

        Args:
            training_result: Result from train_lora or train_dpo.
            key: Unique identifier for the adapter (no slashes).
            description: Optional human-readable description.
            deploy: Whether to deploy the adapter (create symlink).
            overwrite: If True and adapter with same md5 exists, return existing.

        Returns:
            AdapterInfo with registration details.

        Raises:
            ValueError: If adapter with same md5 exists and overwrite=False.
        """
        desc = description or f"{training_result.method} adapter from {training_result.base_model}"
        try:
            return self._storage.store_adapter(training_result, key, desc, deploy)
        except ValueError as e:
            if not overwrite or "already exists" not in str(e):
                raise
            # overwrite=True: return existing adapter info
            md5 = (training_result.adapter.md5 if training_result.adapter else None) or "unknown"
            existing = self._storage.get_adapter_by_md5(key, md5)
            if existing:
                self._lg.info(
                    f"adapter '{key}' already exists with md5 {md5[:8]}, returning existing"
                )
                return existing
            raise

    def refresh(self, key: str | None = None, timeout: float = 10.0) -> dict[str, object]:
        """Notify llm-infer to reload adapters.

        Args:
            key: Specific adapter to refresh, or None for full scan.
            timeout: Request timeout in seconds.

        Returns:
            Response from llm-infer API.
        """
        url = f"{self.infer_url}/v1/adapters/refresh"
        params = {}
        if key:
            self._storage.validate_key(key)
            params["key"] = key

        try:
            response = httpx.post(url, params=params, timeout=timeout)
            response.raise_for_status()
            result: dict[str, object] = response.json()
            self._lg.info(f"refresh response: {result}")
            return result
        except httpx.ConnectError as e:
            self._lg.warning(f"could not connect to llm-infer at {self.infer_url}: {e}")
            raise ConnectionError(f"llm-infer not available at {self.infer_url}") from e
        except httpx.HTTPStatusError as e:
            self._lg.error(f"refresh failed: {e.response.text}")
            raise RuntimeError(f"Refresh failed: {e.response.text}") from e

    def register_and_refresh(
        self,
        training_result: RunResult,
        key: str,
        description: str | None = None,
        deploy: bool = True,
        overwrite: bool = False,
    ) -> AdapterInfo:
        """Register an adapter and immediately refresh llm-infer.

        Convenience method that combines register() and refresh().

        Args:
            training_result: Result from train_lora or train_dpo.
            key: Unique identifier for the adapter.
            description: Optional description.
            deploy: Whether to deploy the adapter.
            overwrite: Ignored (storage handles versioning).

        Returns:
            AdapterInfo with registration details.
        """
        info = self.register(
            training_result=training_result,
            key=key,
            description=description,
            deploy=deploy,
            overwrite=overwrite,
        )
        self.refresh(key=key)
        return info

    # =========================================================================
    # Storage Delegation (thin wrappers for backwards compatibility)
    # =========================================================================

    def list(self) -> list[AdapterInfo]:
        """List all registered adapters."""
        return self._storage.list_adapter_infos()

    def get(self, key: str) -> AdapterInfo | None:
        """Get info for a specific adapter."""
        return self._storage.get_adapter(key)

    def remove(self, key: str, version_id: str | None = None) -> None:
        """Remove an adapter or specific version."""
        self._storage.remove_adapter(key, version_id)

    def set_deployed(self, key: str, deployed: bool, version_id: str | None = None) -> None:
        """Deploy or undeploy an adapter."""
        if deployed:
            self._storage.deploy_adapter(key, version_id)
        else:
            self._storage.undeploy_adapter(key)

    def is_deployed(self, key: str) -> bool:
        """Check if an adapter is deployed."""
        return self._storage.is_deployed(key)
