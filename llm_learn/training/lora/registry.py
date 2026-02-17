"""Adapter registry for llm-infer integration.

Handles:
- Storing trained adapters in registry directory
- Deploying adapters via symlinks to deployed directory
- Writing config.yaml for each adapter
- Calling llm-infer's refresh API

Directory structure:
    base_path/
    ├── registry/      # All adapters stored here
    │   ├── v1/
    │   │   ├── adapter_model.safetensors
    │   │   └── config.yaml
    │   └── v2/
    └── deployed/      # Symlinks to enabled adapters (vLLM scans this)
        └── v1 -> ../registry/v1
"""

import shutil
from dataclasses import dataclass
from pathlib import Path

import httpx
import yaml
from appinfra.log import Logger

from ..config import RunResult


@dataclass
class AdapterInfo:
    """Information about a registered adapter."""

    adapter_id: str
    path: Path
    deployed: bool
    description: str | None = None
    based_on: str | None = None


class AdapterRegistry:
    """Registry for managing LoRA adapters with llm-infer.

    Handles the lifecycle of trained adapters:
    1. Register: Copy adapter files to registry directory
    2. Deploy/Undeploy: Manage symlinks in deployed directory
    3. Refresh: Notify llm-infer to reload adapters
    4. Remove: Delete adapter from registry (and deployed if applicable)
    """

    def __init__(
        self,
        lg: Logger,
        base_path: str | Path,
        infer_url: str = "http://localhost:8000",
    ):
        """Initialize adapter registry.

        Args:
            lg: Logger instance
            base_path: Parent path containing registry/ and deployed/ subdirs
            infer_url: Base URL for llm-infer API
        """
        self._lg = lg
        self.base_path = Path(base_path).expanduser()
        self.registry_path = self.base_path / "registry"
        self.deployed_path = self.base_path / "deployed"
        self.infer_url = infer_url.rstrip("/")

        # Ensure directories exist
        for path in (self.registry_path, self.deployed_path):
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)
                self._lg.info(f"created directory: {path}")

    def _validate_adapter_id(self, adapter_id: str) -> None:
        """Validate adapter_id has no path traversal characters."""
        if "/" in adapter_id or "\\" in adapter_id or ".." in adapter_id:
            raise ValueError(f"Invalid adapter_id: {adapter_id}")

    def _write_adapter_config(
        self,
        adapter_path: Path,
        training_result: RunResult,
        description: str,
    ) -> None:
        """Write config.yaml for the adapter."""
        config: dict = {
            "enabled": True,  # Deployment controlled via symlinks, not this flag
            "description": description,
            "base_model": training_result.base_model,
            "method": training_result.method,
            "training_config": training_result.config,
        }
        if training_result.based_on is not None:
            config["based_on"] = str(training_result.based_on)
        config_path = adapter_path / "config.yaml"
        with config_path.open("w") as f:
            yaml.safe_dump(config, f)
        self._lg.info(f"wrote config: {config_path}")

    def _read_adapter_config(self, adapter_path: Path) -> dict:
        """Read config.yaml for an adapter."""
        config_path = adapter_path / "config.yaml"
        if not config_path.exists():
            return {}
        with config_path.open("r") as f:
            return yaml.safe_load(f) or {}

    def is_deployed(self, adapter_id: str) -> bool:
        """Check if an adapter is deployed (symlink exists).

        Args:
            adapter_id: Adapter to check

        Returns:
            True if adapter is deployed (symlink exists in deployed/)
        """
        self._validate_adapter_id(adapter_id)
        return (self.deployed_path / adapter_id).is_symlink()

    def resolve(self, ref: str | Path) -> Path:
        """Resolve adapter reference (path or ID) to a path.

        Args:
            ref: Either a filesystem path or a registered adapter ID.

        Returns:
            Path to the adapter directory.

        Raises:
            ValueError: If the adapter is not found.
        """
        path = Path(ref)
        if path.exists():
            return path
        if isinstance(ref, str):
            info = self.get(ref)
            if info:
                return info.path
        raise ValueError(f"Adapter not found: {ref}")

    def _prepare_adapter_path(self, adapter_id: str, source_path: Path, overwrite: bool) -> Path:
        """Validate and prepare adapter path, removing existing if overwriting."""
        self._validate_adapter_id(adapter_id)
        adapter_path = self.registry_path / adapter_id

        if adapter_path.exists() and not overwrite:
            raise ValueError(
                f"Adapter '{adapter_id}' already exists. Use overwrite=True to replace."
            )
        if not source_path.exists():
            raise FileNotFoundError(f"Adapter path not found: {source_path}")

        if adapter_path.exists():
            self._remove_deploy_symlink(adapter_id)
            shutil.rmtree(adapter_path)
            self._lg.info(f"removed existing adapter: {adapter_id}")

        return adapter_path

    def register(
        self,
        training_result: RunResult,
        adapter_id: str,
        description: str | None = None,
        deploy: bool = True,
        overwrite: bool = False,
    ) -> AdapterInfo:
        """Register a trained adapter.

        Copies adapter files to the registry directory.

        Args:
            training_result: Result from train_lora or train_dpo
            adapter_id: Unique identifier for the adapter (no slashes)
            description: Optional human-readable description
            deploy: Whether to deploy the adapter (create symlink)
            overwrite: If True, overwrite existing adapter with same ID

        Returns:
            AdapterInfo with registration details

        Raises:
            ValueError: If adapter_id is invalid or already exists
            FileNotFoundError: If training result path doesn't exist
        """
        adapter_path = self._prepare_adapter_path(
            adapter_id, training_result.adapter_path, overwrite
        )
        shutil.copytree(training_result.adapter_path, adapter_path)
        self._lg.info(f"copied adapter to: {adapter_path}")

        desc = description or f"{training_result.method} adapter from {training_result.base_model}"
        self._write_adapter_config(adapter_path, training_result, desc)

        if deploy:
            self._create_deploy_symlink(adapter_id)

        config = self._read_adapter_config(adapter_path)
        return AdapterInfo(
            adapter_id=adapter_id,
            path=adapter_path,
            deployed=deploy,
            description=desc,
            based_on=config.get("based_on"),
        )

    def _create_deploy_symlink(self, adapter_id: str) -> None:
        """Create symlink in deployed directory."""
        symlink_path = self.deployed_path / adapter_id

        if symlink_path.exists() or symlink_path.is_symlink():
            symlink_path.unlink()

        # Use relative path for symlink
        relative_target = Path("..") / "registry" / adapter_id
        symlink_path.symlink_to(relative_target)
        self._lg.info(f"created symlink: {symlink_path} -> {relative_target}")

    def _remove_deploy_symlink(self, adapter_id: str) -> None:
        """Remove symlink from deployed directory if it exists."""
        symlink_path = self.deployed_path / adapter_id
        if symlink_path.is_symlink():
            symlink_path.unlink()
            self._lg.info(f"removed symlink: {symlink_path}")

    def set_deployed(self, adapter_id: str, deployed: bool) -> None:
        """Deploy or undeploy an adapter.

        Args:
            adapter_id: Adapter to modify
            deployed: True to deploy (create symlink), False to undeploy
        """
        self._validate_adapter_id(adapter_id)
        adapter_path = self.registry_path / adapter_id

        if not adapter_path.exists():
            raise ValueError(f"Adapter '{adapter_id}' not found")

        if deployed:
            self._create_deploy_symlink(adapter_id)
        else:
            self._remove_deploy_symlink(adapter_id)

        self._lg.info(f"set adapter '{adapter_id}' deployed={deployed}")

    def remove(self, adapter_id: str) -> None:
        """Remove an adapter from the registry.

        Args:
            adapter_id: Adapter to remove
        """
        self._validate_adapter_id(adapter_id)
        adapter_path = self.registry_path / adapter_id

        if not adapter_path.exists():
            raise ValueError(f"Adapter '{adapter_id}' not found")

        # Remove symlink first if deployed
        self._remove_deploy_symlink(adapter_id)

        # Remove adapter directory
        shutil.rmtree(adapter_path)
        self._lg.info(f"removed adapter: {adapter_id}")

    def list(self) -> list[AdapterInfo]:
        """List all registered adapters.

        Returns:
            List of AdapterInfo for each adapter
        """
        adapters = []
        for path in self.registry_path.iterdir():
            if not path.is_dir():
                continue

            config_path = path / "config.yaml"
            if not config_path.exists():
                continue

            config = self._read_adapter_config(path)
            adapter_id = path.name

            adapters.append(
                AdapterInfo(
                    adapter_id=adapter_id,
                    path=path,
                    deployed=self.is_deployed(adapter_id),
                    description=config.get("description"),
                    based_on=config.get("based_on"),
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
        self._validate_adapter_id(adapter_id)
        adapter_path = self.registry_path / adapter_id
        config_path = adapter_path / "config.yaml"

        if not config_path.exists():
            return None

        config = self._read_adapter_config(adapter_path)

        return AdapterInfo(
            adapter_id=adapter_id,
            path=adapter_path,
            deployed=self.is_deployed(adapter_id),
            description=config.get("description"),
            based_on=config.get("based_on"),
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
            self._validate_adapter_id(adapter_id)
            params["adapter_id"] = adapter_id

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
        adapter_id: str,
        description: str | None = None,
        deploy: bool = True,
        overwrite: bool = False,
    ) -> AdapterInfo:
        """Register an adapter and immediately refresh llm-infer.

        Convenience method that combines register() and refresh().

        Args:
            training_result: Result from train_lora or train_dpo
            adapter_id: Unique identifier for the adapter
            description: Optional description
            deploy: Whether to deploy the adapter
            overwrite: If True, overwrite existing adapter

        Returns:
            AdapterInfo with registration details
        """
        info = self.register(
            training_result=training_result,
            adapter_id=adapter_id,
            description=description,
            deploy=deploy,
            overwrite=overwrite,
        )
        self.refresh(adapter_id=adapter_id)
        return info
