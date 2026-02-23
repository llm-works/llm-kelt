"""Adapter registry for llm-infer integration.

Handles:
- Storing trained adapters in adapters directory
- Deploying adapters via symlinks to deployed directory
- Writing config.yaml for each adapter
- Calling llm-infer's refresh API

Directory structure:
    base_path/
    ├── pending/       # Manifests waiting for training
    │   └── coding-v1.yaml
    ├── completed/     # Finished manifests (archive)
    │   └── coding-v0.yaml
    ├── adapters/      # All adapters stored here
    │   ├── coding-v1/
    │   │   ├── adapter_model.safetensors
    │   │   └── config.yaml
    │   └── writing-v1/
    └── deployed/      # Symlinks to enabled adapters (vLLM scans this)
        └── coding-v1 -> ../adapters/coding-v1
"""

import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import httpx
import yaml
from appinfra.log import Logger

from ..schema import RunResult


def _make_version_id(md5: str, timestamp: datetime | None = None) -> str:
    """Generate version ID: YYYYMMDD-HHMMSS-{md5}."""
    ts = timestamp or datetime.now()
    return f"{ts.strftime('%Y%m%d-%H%M%S')}-{md5}"


@dataclass
class AdapterInfo:
    """Information about a registered adapter."""

    key: str
    path: Path
    deployed: bool
    description: str | None = None
    parent: str | None = None


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
            base_path: Parent path containing adapters/ and deployed/ subdirs
            infer_url: Base URL for llm-infer API
        """
        self._lg = lg
        self.base_path = Path(base_path).expanduser()
        self.adapters_path = self.base_path / "adapters"
        self.deployed_path = self.base_path / "deployed"
        self.infer_url = infer_url.rstrip("/")

        # Ensure directories exist
        for path in (self.adapters_path, self.deployed_path):
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)
                self._lg.info(f"created directory: {path}")

    def _validate_key(self, key: str) -> None:
        """Validate key has no path traversal characters."""
        if "/" in key or "\\" in key or ".." in key:
            raise ValueError(f"Invalid key: {key}")

    def _write_adapter_config(
        self,
        adapter_path: Path,
        training_result: RunResult,
        description: str,
    ) -> None:
        """Write config.yaml for the adapter."""
        # Convert DotDict to plain dict for YAML serialization
        training_config = json.loads(json.dumps(training_result.config))
        config: dict = {
            "enabled": True,  # Deployment controlled via symlinks, not this flag
            "description": description,
            "base_model": training_result.base_model,
            "method": training_result.method,
            "training_config": training_config,
        }
        parent = training_result.get("parent")
        if parent is not None:
            config["parent"] = parent.path
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

    def is_deployed(self, key: str) -> bool:
        """Check if an adapter is deployed (symlink exists).

        Args:
            key: Adapter to check

        Returns:
            True if adapter is deployed (symlink exists in deployed/)
        """
        self._validate_key(key)
        return (self.deployed_path / key).is_symlink()

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

    def _prepare_adapter_path(
        self, key: str, md5: str, source_path: Path, *, overwrite: bool = False
    ) -> tuple[Path, str]:
        """Validate and prepare adapter path (versioned by timestamp-md5).

        Returns:
            Tuple of (adapter_path, version_id).

        Raises:
            ValueError: If adapter key already exists and overwrite is False.
        """
        self._validate_key(key)
        if not source_path.exists():
            raise FileNotFoundError(f"Adapter path not found: {source_path}")

        # Check if key already has any versions (for overwrite check)
        key_path = self.adapters_path / key
        if key_path.exists() and any(key_path.iterdir()) and not overwrite:
            raise ValueError(f"Adapter '{key}' already exists. Use overwrite=True to replace.")

        # Structure: adapters/{key}/{YYYYMMDD-HHMMSS-md5}/
        version_id = _make_version_id(md5)
        adapter_path = key_path / version_id
        if adapter_path.exists():
            self._lg.info(f"adapter version already exists: {adapter_path}")
            shutil.rmtree(adapter_path)

        return adapter_path, version_id

    def register(
        self,
        training_result: RunResult,
        key: str,
        description: str | None = None,
        deploy: bool = True,
        overwrite: bool = False,
    ) -> AdapterInfo:
        """Register a trained adapter.

        Copies adapter files to the registry directory.

        Args:
            training_result: Result from train_lora or train_dpo
            key: Unique identifier for the adapter (no slashes)
            description: Optional human-readable description
            deploy: Whether to deploy the adapter (create symlink)
            overwrite: If True, overwrite existing adapter with same ID

        Returns:
            AdapterInfo with registration details

        Raises:
            ValueError: If key is invalid or already exists
            FileNotFoundError: If training result path doesn't exist
        """
        if training_result.adapter is None:
            raise ValueError("Training result has no adapter")
        source_path = Path(training_result.adapter.path)
        md5 = training_result.adapter.md5
        adapter_path, version_id = self._prepare_adapter_path(
            key, md5, source_path, overwrite=overwrite
        )
        shutil.copytree(source_path, adapter_path)
        self._lg.info(f"copied adapter to: {adapter_path}")

        desc = description or f"{training_result.method} adapter from {training_result.base_model}"
        self._write_adapter_config(adapter_path, training_result, desc)

        if deploy:
            self._create_deploy_symlink(key, version_id)

        config = self._read_adapter_config(adapter_path)
        return AdapterInfo(
            key=key,
            path=adapter_path,
            deployed=deploy,
            description=desc,
            parent=config.get("parent"),
        )

    def _create_deploy_symlink(self, key: str, version_id: str | None = None) -> None:
        """Create symlink in deployed directory to specific version."""
        symlink_path = self.deployed_path / key

        if symlink_path.exists() or symlink_path.is_symlink():
            symlink_path.unlink()

        # Resolve latest version if not specified
        if not version_id:
            latest_path = self._get_latest_version_path(key)
            if latest_path:
                version_id = latest_path.name

        # Use relative path for symlink: deployed/key -> ../adapters/key/version_id
        if version_id:
            relative_target = Path("..") / "adapters" / key / version_id
        else:
            # Fallback to directory (shouldn't happen if adapter exists)
            relative_target = Path("..") / "adapters" / key
        symlink_path.symlink_to(relative_target)
        self._lg.info(f"created symlink: {symlink_path} -> {relative_target}")

    def _remove_deploy_symlink(self, key: str) -> None:
        """Remove symlink from deployed directory if it exists."""
        symlink_path = self.deployed_path / key
        if symlink_path.is_symlink():
            symlink_path.unlink()
            self._lg.info(f"removed symlink: {symlink_path}")

    def set_deployed(self, key: str, deployed: bool, version_id: str | None = None) -> None:
        """Deploy or undeploy an adapter.

        Args:
            key: Adapter to modify
            deployed: True to deploy (create symlink), False to undeploy
            version_id: Specific version to deploy (uses latest if not specified)
        """
        self._validate_key(key)
        key_path = self.adapters_path / key

        if not key_path.exists():
            raise ValueError(f"Adapter '{key}' not found")

        if deployed:
            if version_id is None:
                # Find latest version
                version_path = self._get_latest_version_path(key)
                if version_path is None:
                    raise ValueError(f"No versions found for adapter '{key}'")
                version_id = version_path.name
            self._create_deploy_symlink(key, version_id)
        else:
            self._remove_deploy_symlink(key)

        self._lg.info(f"set adapter '{key}' deployed={deployed}")

    def remove(self, key: str, version_id: str | None = None) -> None:
        """Remove an adapter (all versions) or specific version.

        If the removed version is currently deployed, the symlink is also removed,
        leaving the adapter undeployed. Call set_deployed() to deploy another version.

        Args:
            key: Adapter to remove
            version_id: Specific version to remove (removes all versions if not specified)
        """
        self._validate_key(key)
        key_path = self.adapters_path / key

        if not key_path.exists():
            raise ValueError(f"Adapter '{key}' not found")

        if version_id:
            # Remove specific version
            version_path = key_path / version_id
            if not version_path.exists():
                raise ValueError(f"Version '{version_id}' not found for adapter '{key}'")
            # If this version is deployed, remove symlink
            deployed_path = self._get_deployed_version_path(key)
            if deployed_path and deployed_path == version_path:
                self._remove_deploy_symlink(key)
            shutil.rmtree(version_path)
            self._lg.info(f"removed adapter version: {key}/{version_id}")
        else:
            # Remove all versions
            self._remove_deploy_symlink(key)
            shutil.rmtree(key_path)
            self._lg.info(f"removed adapter: {key}")

    def _get_deployed_version_path(self, key: str) -> Path | None:
        """Get path to deployed version of an adapter."""
        symlink = self.deployed_path / key
        if symlink.is_symlink():
            return symlink.resolve()
        return None

    def _get_latest_version_path(self, key: str) -> Path | None:
        """Get path to latest version of an adapter (by version ID).

        Version IDs are YYYYMMDD-HHMMSS-{md5}, so lexicographic sort by name
        gives chronological order (immune to mtime changes from file copies).
        """
        key_path = self.adapters_path / key
        if not key_path.is_dir():
            return None
        versions = [p for p in key_path.iterdir() if p.is_dir() and (p / "config.yaml").exists()]
        if not versions:
            return None
        return max(versions, key=lambda p: p.name)

    def list(self) -> list[AdapterInfo]:
        """List all registered adapters (shows deployed or latest version)."""
        adapters = []
        for key_path in self.adapters_path.iterdir():
            if not key_path.is_dir():
                continue

            key = key_path.name
            version_path = self._get_deployed_version_path(key) or self._get_latest_version_path(
                key
            )
            if version_path is None:
                continue

            config = self._read_adapter_config(version_path)
            adapters.append(
                AdapterInfo(
                    key=key,
                    path=version_path,
                    deployed=self.is_deployed(key),
                    description=config.get("description"),
                    parent=config.get("parent"),
                )
            )

        return adapters

    def get(self, key: str) -> AdapterInfo | None:
        """Get info for a specific adapter (deployed or latest version)."""
        self._validate_key(key)
        version_path = self._get_deployed_version_path(key) or self._get_latest_version_path(key)
        if version_path is None:
            return None

        config = self._read_adapter_config(version_path)
        return AdapterInfo(
            key=key,
            path=version_path,
            deployed=self.is_deployed(key),
            description=config.get("description"),
            parent=config.get("parent"),
        )

    def refresh(self, key: str | None = None, timeout: float = 10.0) -> dict[str, object]:
        """Notify llm-infer to reload adapters.

        Args:
            key: Specific adapter to refresh, or None for full scan
            timeout: Request timeout in seconds

        Returns:
            Response from llm-infer API
        """
        url = f"{self.infer_url}/v1/adapters/refresh"
        params = {}
        if key:
            self._validate_key(key)
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
            training_result: Result from train_lora or train_dpo
            key: Unique identifier for the adapter
            description: Optional description
            deploy: Whether to deploy the adapter
            overwrite: If True, overwrite existing adapter

        Returns:
            AdapterInfo with registration details
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
