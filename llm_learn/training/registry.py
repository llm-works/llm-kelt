"""Adapter registry for llm-infer integration.

Handles:
- Copying trained adapters to llm-infer's adapter directory
- Writing config.yaml for each adapter
- Calling llm-infer's refresh API
"""

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

import httpx
import yaml

from .config import TrainingResult

logger = logging.getLogger(__name__)


@dataclass
class AdapterInfo:
    """Information about a registered adapter."""

    adapter_id: str
    path: Path
    enabled: bool
    description: str | None = None


class AdapterRegistry:
    """Registry for managing LoRA adapters with llm-infer.

    Handles the lifecycle of trained adapters:
    1. Register: Copy adapter files to llm-infer's base path
    2. Enable/Disable: Update config.yaml
    3. Refresh: Notify llm-infer to reload adapters
    4. Remove: Delete adapter from llm-infer's base path
    """

    def __init__(
        self,
        base_path: str | Path,
        infer_url: str = "http://localhost:8000",
    ):
        """Initialize adapter registry.

        Args:
            base_path: Path to llm-infer's adapter directory
            infer_url: Base URL for llm-infer API
        """
        self.base_path = Path(base_path).expanduser()
        self.infer_url = infer_url.rstrip("/")

        if not self.base_path.exists():
            self.base_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created adapter base path: {self.base_path}")

    def _validate_adapter_id(self, adapter_id: str) -> None:
        """Validate adapter_id has no path traversal characters."""
        if "/" in adapter_id or "\\" in adapter_id or ".." in adapter_id:
            raise ValueError(f"Invalid adapter_id: {adapter_id}")

    def _write_adapter_config(
        self,
        adapter_path: Path,
        training_result: TrainingResult,
        description: str,
        enabled: bool,
    ) -> None:
        """Write config.yaml for the adapter."""
        config = {
            "enabled": enabled,
            "description": description,
            "base_model": training_result.base_model,
            "method": training_result.method,
            "training_config": training_result.config,
        }
        config_path = adapter_path / "config.yaml"
        with config_path.open("w") as f:
            yaml.safe_dump(config, f)
        logger.info(f"Wrote config: {config_path}")

    def register(
        self,
        training_result: TrainingResult,
        adapter_id: str,
        description: str | None = None,
        enabled: bool = True,
        overwrite: bool = False,
    ) -> AdapterInfo:
        """Register a trained adapter with llm-infer.

        Copies adapter files to the base path and creates config.yaml.

        Args:
            training_result: Result from train_lora or train_dpo
            adapter_id: Unique identifier for the adapter (no slashes)
            description: Optional human-readable description
            enabled: Whether the adapter should be enabled
            overwrite: If True, overwrite existing adapter with same ID

        Returns:
            AdapterInfo with registration details

        Raises:
            ValueError: If adapter_id is invalid or already exists
            FileNotFoundError: If training result path doesn't exist
        """
        self._validate_adapter_id(adapter_id)
        adapter_path = self.base_path / adapter_id

        if adapter_path.exists() and not overwrite:
            raise ValueError(
                f"Adapter '{adapter_id}' already exists. Use overwrite=True to replace."
            )
        if not training_result.adapter_path.exists():
            raise FileNotFoundError(f"Adapter path not found: {training_result.adapter_path}")

        # Remove existing if overwriting
        if adapter_path.exists():
            shutil.rmtree(adapter_path)
            logger.info(f"Removed existing adapter: {adapter_id}")

        # Copy adapter files
        shutil.copytree(training_result.adapter_path, adapter_path)
        logger.info(f"Copied adapter to: {adapter_path}")

        desc = description or f"{training_result.method} adapter from {training_result.base_model}"
        self._write_adapter_config(adapter_path, training_result, desc, enabled)

        return AdapterInfo(
            adapter_id=adapter_id,
            path=adapter_path,
            enabled=enabled,
            description=desc,
        )

    def set_enabled(self, adapter_id: str, enabled: bool) -> None:
        """Enable or disable an adapter.

        Args:
            adapter_id: Adapter to modify
            enabled: New enabled state
        """
        config_path = self.base_path / adapter_id / "config.yaml"
        if not config_path.exists():
            raise ValueError(f"Adapter '{adapter_id}' not found")

        with config_path.open("r") as f:
            config = yaml.safe_load(f)

        config["enabled"] = enabled

        with config_path.open("w") as f:
            yaml.safe_dump(config, f)

        logger.info(f"Set adapter '{adapter_id}' enabled={enabled}")

    def remove(self, adapter_id: str) -> None:
        """Remove an adapter from the registry.

        Args:
            adapter_id: Adapter to remove
        """
        adapter_path = self.base_path / adapter_id
        if not adapter_path.exists():
            raise ValueError(f"Adapter '{adapter_id}' not found")

        shutil.rmtree(adapter_path)
        logger.info(f"Removed adapter: {adapter_id}")

    def list(self) -> list[AdapterInfo]:
        """List all registered adapters.

        Returns:
            List of AdapterInfo for each adapter
        """
        adapters = []
        for path in self.base_path.iterdir():
            if not path.is_dir():
                continue

            config_path = path / "config.yaml"
            if not config_path.exists():
                continue

            with config_path.open("r") as f:
                config = yaml.safe_load(f) or {}

            adapters.append(
                AdapterInfo(
                    adapter_id=path.name,
                    path=path,
                    enabled=bool(config.get("enabled", True)),
                    description=str(config["description"]) if config.get("description") else None,
                )
            )

        return adapters

    def get(self, adapter_id: str) -> AdapterInfo | None:
        """Get info for a specific adapter.

        Args:
            adapter_id: Adapter to look up

        Returns:
            AdapterInfo or None if not found
        """
        adapter_path = self.base_path / adapter_id
        config_path = adapter_path / "config.yaml"

        if not config_path.exists():
            return None

        with config_path.open("r") as f:
            config = yaml.safe_load(f) or {}

        return AdapterInfo(
            adapter_id=adapter_id,
            path=adapter_path,
            enabled=bool(config.get("enabled", True)),
            description=str(config["description"]) if config.get("description") else None,
        )

    def refresh(self, adapter_id: str | None = None, timeout: float = 10.0) -> dict[str, object]:
        """Notify llm-infer to reload adapters.

        Args:
            adapter_id: Specific adapter to refresh, or None for full scan
            timeout: Request timeout in seconds

        Returns:
            Response from llm-infer API
        """
        url = f"{self.infer_url}/v1/adapters/refresh"
        params = {}
        if adapter_id:
            params["adapter_id"] = adapter_id

        try:
            response = httpx.post(url, params=params, timeout=timeout)
            response.raise_for_status()
            result: dict[str, object] = response.json()
            logger.info(f"Refresh response: {result}")
            return result
        except httpx.ConnectError as e:
            logger.warning(f"Could not connect to llm-infer at {self.infer_url}: {e}")
            raise ConnectionError(f"llm-infer not available at {self.infer_url}") from e
        except httpx.HTTPStatusError as e:
            logger.error(f"Refresh failed: {e.response.text}")
            raise RuntimeError(f"Refresh failed: {e.response.text}") from e

    def register_and_refresh(
        self,
        training_result: TrainingResult,
        adapter_id: str,
        description: str | None = None,
        enabled: bool = True,
        overwrite: bool = False,
    ) -> AdapterInfo:
        """Register an adapter and immediately refresh llm-infer.

        Convenience method that combines register() and refresh().

        Args:
            training_result: Result from train_lora or train_dpo
            adapter_id: Unique identifier for the adapter
            description: Optional description
            enabled: Whether to enable the adapter
            overwrite: If True, overwrite existing adapter

        Returns:
            AdapterInfo with registration details
        """
        info = self.register(
            training_result=training_result,
            adapter_id=adapter_id,
            description=description,
            enabled=enabled,
            overwrite=overwrite,
        )
        self.refresh(adapter_id=adapter_id)
        return info
