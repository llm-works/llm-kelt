"""File-based storage implementation.

Implements the Storage interface using the local filesystem.
"""

from __future__ import annotations

import gzip
import json
import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import yaml
from appinfra.log import Logger

from .base import Storage

# Pre-compiled regex for key validation (used on hot path)
_KEY_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]*[a-zA-Z0-9]$|^[a-zA-Z0-9]$")

if TYPE_CHECKING:
    from ..manifest.schema import Manifest
    from ..schema import AdapterInfo, RunResult, SubmitResult


class FileStorage(Storage):
    """File-based storage for training artifacts.

    Implements the Storage interface using local filesystem:
    - Manifests stored as YAML files (gzipped for completed)
    - Adapters stored in versioned directories
    - Deployment via symlinks
    - Work areas as temporary directories

    Directory structure:
        base_path/
        ├── pending/       # Manifests waiting for training
        ├── completed/     # Finished manifests (archive)
        ├── adapters/      # All adapters stored here
        │   └── {key}/{version_id}/
        ├── deployed/      # Symlinks to enabled adapters
        └── work/          # Temporary training directories
    """

    def __init__(self, lg: Logger, base_path: str | Path) -> None:
        """Initialize file storage.

        Args:
            lg: Logger instance.
            base_path: Root directory for the registry.
        """
        self._lg = lg
        self.base_path = Path(base_path).expanduser()
        self.pending_path = self.base_path / "pending"
        self.completed_path = self.base_path / "completed"
        self.adapters_path = self.base_path / "adapters"
        self.deployed_path = self.base_path / "deployed"
        self.work_path = self.base_path / "work"

    # =========================================================================
    # Manifest Operations (ABC implementation)
    # =========================================================================

    def submit_manifest(self, manifest: Manifest) -> SubmitResult:
        """Submit a manifest to the pending queue."""
        from ..manifest.loader import save_manifest
        from ..schema import SubmitResult

        self.validate_key(manifest.adapter)
        self.pending_path.mkdir(parents=True, exist_ok=True)

        pending_file = self.pending_path / f"{manifest.adapter}.yaml"
        if pending_file.exists():
            raise ValueError(f"Manifest already pending for adapter: {manifest.adapter}")

        save_manifest(manifest, pending_file)
        self._lg.info("submitted manifest", extra={"adapter": manifest.adapter})

        return SubmitResult(
            adapter=manifest.adapter,
            timestamp=datetime.now(),
            location=str(pending_file),
        )

    def get_pending_manifest(self, adapter: str) -> Manifest | None:
        """Get a pending manifest by adapter key."""
        from ..manifest.loader import load_manifest

        self.validate_key(adapter)
        pending_file = self.pending_path / f"{adapter}.yaml"
        if not pending_file.exists():
            return None
        return load_manifest(pending_file)

    def list_pending_manifests(self) -> list[Manifest]:
        """List all pending manifests."""
        from ..manifest.loader import load_manifest

        if not self.pending_path.exists():
            return []

        manifests = []
        for path in sorted(self.pending_path.glob("*.yaml")):
            try:
                manifests.append(load_manifest(path))
            except Exception as e:
                self._lg.warning(
                    "failed to load manifest", extra={"path": str(path), "error": str(e)}
                )
        return manifests

    def remove_pending_manifest(self, adapter: str) -> None:
        """Remove a manifest from pending queue."""
        self.validate_key(adapter)
        pending_file = self.pending_path / f"{adapter}.yaml"
        if not pending_file.exists():
            raise FileNotFoundError(f"Manifest not found: {adapter}")
        pending_file.unlink()

    def _link_manifest_to_adapter(self, key: str, md5: str, manifest_file: Path) -> None:
        """Create symlink from adapter directory to its completed manifest."""
        version_id = self._find_version_by_md5(key, md5)
        if not version_id:
            return
        adapter_dir = self.adapters_path / key / version_id
        manifest_link = adapter_dir / "manifest"
        if not manifest_link.exists():
            rel_path = os.path.relpath(manifest_file, adapter_dir)
            manifest_link.symlink_to(rel_path)

    def complete_manifest(self, manifest: Manifest) -> None:
        """Move a manifest to completed storage."""
        from ..manifest.loader import save_manifest

        self.completed_path.mkdir(parents=True, exist_ok=True)

        md5 = manifest.output.adapter.md5 if manifest.output and manifest.output.adapter else None
        # Use timestamp suffix when md5 is unavailable to avoid overwriting
        if md5:
            suffix = md5
        else:
            suffix = datetime.now().strftime("%Y%m%d-%H%M%S")
            self._lg.warning(
                "completing manifest without md5, using timestamp",
                extra={"adapter": manifest.adapter},
            )
        completed_file = self.completed_path / f"{manifest.adapter}-{suffix}.yaml.gz"
        save_manifest(manifest, completed_file, compress=True)

        if md5:
            self._link_manifest_to_adapter(manifest.adapter, md5, completed_file)

        # Remove from pending if it exists
        pending_file = self.pending_path / f"{manifest.adapter}.yaml"
        if pending_file.exists():
            pending_file.unlink()

        self._lg.info("completed manifest", extra={"adapter": manifest.adapter, "md5": md5})

    def list_completed_manifests(self) -> list[Manifest]:
        """List all completed manifests."""
        from ..manifest.loader import load_manifest

        if not self.completed_path.exists():
            return []

        manifests = []
        # Use explicit patterns to avoid matching editor temp files (.yaml.bak, etc.)
        paths = list(self.completed_path.glob("*.yaml")) + list(
            self.completed_path.glob("*.yaml.gz")
        )
        for path in sorted(paths):
            try:
                manifests.append(load_manifest(path))
            except Exception as e:
                self._lg.warning(
                    "failed to load manifest", extra={"path": str(path), "error": str(e)}
                )
        return manifests

    def _normalize_md5(self, md5: object) -> str | None:
        """Normalize and validate MD5 input. Returns 12-char hex or None if invalid."""
        if not isinstance(md5, str):
            self._lg.warning("invalid md5 type", extra={"md5": md5, "type": type(md5).__name__})
            return None
        md5 = md5.lower()
        if re.fullmatch(r"[a-f0-9]{32}", md5):
            return md5[:12]  # Truncate full MD5 to 12-char prefix used in filenames
        if re.fullmatch(r"[a-f0-9]{12}", md5):
            return md5
        self._lg.warning("invalid md5 format", extra={"md5": md5})
        return None

    def find_adapter_by_md5(self, md5: str) -> Manifest | None:
        """Find a completed manifest by adapter MD5.

        Uses filename glob (*-{md5}.yaml.gz) for O(1) lookup, and streams only
        metadata (stops before data records) for fast lineage lookups.

        Args:
            md5: Hex MD5 hash (12 or 32 chars). Invalid input returns None.
        """
        from ..manifest.loader import load_manifest_metadata

        if not self.completed_path.exists():
            return None
        if (normalized := self._normalize_md5(md5)) is None:
            return None

        # md5 is in filename: {adapter}-{md5}.yaml.gz
        matches = sorted(self.completed_path.glob(f"*-{normalized}.yaml.gz"))
        if not matches:
            matches = sorted(self.completed_path.glob(f"*-{normalized}.yaml"))
        if not matches:
            return None

        try:
            return load_manifest_metadata(matches[0])
        except Exception as e:
            self._lg.warning(
                "failed to load manifest", extra={"path": str(matches[0]), "error": str(e)}
            )
            return None

    # =========================================================================
    # Adapter Operations (ABC implementation)
    # =========================================================================

    def store_adapter(
        self,
        training_result: RunResult,
        key: str,
        description: str,
        deploy: bool | Literal["add", "replace"] = True,
    ) -> AdapterInfo:
        """Store a trained adapter (ABC-compliant signature).

        Args:
            training_result: Result from training with adapter path.
            key: Adapter key.
            description: Human-readable description.
            deploy: Deployment setting (True/"replace", "add", or False).

        Returns:
            AdapterInfo with registration details.
        """
        return self._store_adapter_from_result(training_result, key, description, deploy)

    def _validate_training_result(self, training_result: RunResult) -> Path:
        """Validate training result and return source path."""
        if not training_result.adapter or not training_result.adapter.path:
            raise ValueError("Training result has no adapter path")

        source = Path(training_result.adapter.path)
        if not source.exists():
            raise FileNotFoundError(f"Adapter source not found: {source}")
        return source

    def _write_adapter_result_config(
        self, adapter_path: Path, key: str, version_id: str, description: str, result: RunResult
    ) -> None:
        """Write config.yaml for adapter from training result."""
        md5 = (result.adapter.md5 if result.adapter else None) or "unknown"
        config = {
            "key": key,
            "version_id": version_id,
            "description": description,
            "md5": md5,
            "mtime": result.adapter.mtime if result.adapter else None,
            "created_at": datetime.now().isoformat(),
            "samples_trained": result.samples_trained,
            "duration_seconds": result.duration_seconds,
        }
        config_path = adapter_path / "config.yaml"
        with config_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(config, f)

    def _find_version_by_md5(self, key: str, md5: str) -> str | None:
        """Find existing version with matching md5."""
        key_path = self.adapters_path / key
        if not key_path.is_dir():
            return None

        for version_dir in key_path.iterdir():
            if not version_dir.is_dir():
                continue
            config = self._read_adapter_config(key, version_dir.name)
            if config.get("md5") == md5:
                return version_dir.name
        return None

    def get_adapter_by_md5(self, key: str, md5: str) -> AdapterInfo | None:
        """Get adapter info by key and md5."""
        from ..schema import AdapterInfo

        self.validate_key(key)
        version_id = self._find_version_by_md5(key, md5)
        if version_id is None:
            return None

        config = self._read_adapter_config(key, version_id)
        adapter_md5 = config.get("md5")
        return AdapterInfo(
            key=key,
            version_id=version_id,
            path=str(self.adapters_path / key / version_id),
            description=config.get("description", ""),
            md5=adapter_md5,
            deployed=self.is_deployed(key, adapter_md5),
        )

    def _check_duplicate_md5(self, key: str, md5: str) -> None:
        """Raise if adapter with same md5 already exists."""
        from .base import DupAdapterError

        existing_version = self._find_version_by_md5(key, md5)
        if existing_version:
            raise DupAdapterError(key, md5)

    def _copy_adapter_files(self, source: Path, key: str, version_id: str) -> Path:
        """Copy adapter files to registry and return path."""
        adapter_path = self.adapters_path / key / version_id
        if adapter_path.exists():
            shutil.rmtree(adapter_path)
        shutil.copytree(source, adapter_path)
        return adapter_path

    def _store_adapter_from_result(
        self,
        training_result: RunResult,
        key: str,
        description: str,
        deploy: bool | Literal["add", "replace"] = True,
    ) -> AdapterInfo:
        """Store adapter from training result (ABC API).

        Args:
            training_result: Result from training with adapter path.
            key: Adapter key.
            description: Human-readable description.
            deploy: Deployment setting:
                - True or "replace": Deploy and remove existing {key}-* symlinks.
                - "add": Deploy and keep existing {key}-* symlinks.
                - False: Don't deploy.

        Raises:
            ValueError: If adapter with same md5 already exists for this key.
        """
        from ..schema import AdapterInfo

        self.validate_key(key)
        source = self._validate_training_result(training_result)
        # _validate_training_result guarantees adapter is not None
        assert training_result.adapter is not None
        md5 = training_result.adapter.md5 or "unknown"

        # Only check for duplicates if we have a real md5 hash
        if md5 != "unknown":
            self._check_duplicate_md5(key, md5)

        version_id = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{md5}"
        adapter_path = self._copy_adapter_files(source, key, version_id)

        self._write_adapter_result_config(
            adapter_path, key, version_id, description, training_result
        )
        self._lg.info("stored adapter", extra={"key": key, "version": version_id})

        deployed = self._handle_deploy(key, version_id, deploy)
        return AdapterInfo(
            key=key,
            version_id=version_id,
            path=str(adapter_path),
            description=description,
            md5=md5,
            deployed=deployed,
        )

    def _handle_deploy(
        self, key: str, version_id: str, deploy: bool | Literal["add", "replace"]
    ) -> bool:
        """Handle deployment based on deploy parameter."""
        if deploy is False:
            return False
        policy: Literal["add", "replace"] = deploy if isinstance(deploy, str) else "replace"
        self.deploy_adapter(key, version_id, policy=policy)
        return True

    def get_adapter(self, key: str) -> AdapterInfo | None:
        """Get adapter info by key."""
        from ..schema import AdapterInfo

        self.validate_key(key)
        key_path = self.adapters_path / key
        if not key_path.exists():
            return None

        version_id = self._get_latest_version(key)
        if not version_id:
            return None

        config = self._read_adapter_config(key, version_id)
        adapter_md5 = config.get("md5")
        return AdapterInfo(
            key=key,
            version_id=version_id,
            path=str(key_path / version_id),
            description=config.get("description", ""),
            md5=adapter_md5,
            deployed=self.is_deployed(key, adapter_md5),
        )

    def list_adapters(self) -> list[str]:  # type: ignore[override]
        """List all registered adapter keys.

        Returns strings for backwards compatibility.
        Use list_adapter_infos() for full AdapterInfo objects.
        """
        if not self.adapters_path.exists():
            return []
        return [p.name for p in self.adapters_path.iterdir() if p.is_dir()]

    def list_adapter_infos(self) -> list[AdapterInfo]:
        """List all registered adapters with full info (ABC-compliant)."""
        from ..schema import AdapterInfo

        if not self.adapters_path.exists():
            return []

        adapters = []
        for key_path in sorted(self.adapters_path.iterdir()):
            if not key_path.is_dir():
                continue
            key = key_path.name
            version_id = self._get_latest_version(key)
            if not version_id:
                continue

            config = self._read_adapter_config(key, version_id)
            # md5 is always present - we don't support legacy configs without it
            adapter_md5 = config.get("md5")
            adapters.append(
                AdapterInfo(
                    key=key,
                    version_id=version_id,
                    path=str(key_path / version_id),
                    description=config.get("description", ""),
                    md5=adapter_md5,
                    deployed=self.is_deployed(key, adapter_md5),
                )
            )
        return adapters

    def remove_adapter(self, key: str, version_id: str | None = None) -> None:
        """Remove an adapter or specific version."""
        self.validate_key(key)
        key_path = self.adapters_path / key

        if not key_path.exists():
            raise ValueError(f"Adapter '{key}' not found")

        # Undeploy if removing entire adapter
        if version_id is None and self.is_deployed(key):
            self.undeploy_adapter(key)

        if version_id:
            version_path = key_path / version_id
            if not version_path.exists():
                raise ValueError(f"Version '{version_id}' not found for adapter '{key}'")
            # Undeploy only this specific version if deployed
            config = self._read_adapter_config(key, version_id)
            version_md5 = config.get("md5")
            if version_md5 and self.is_deployed(key, version_md5):
                self.undeploy_adapter(key, version_md5)
            shutil.rmtree(version_path)
            self._lg.info("removed adapter version", extra={"key": key, "version": version_id})
        else:
            shutil.rmtree(key_path)
            self._lg.info("removed adapter", extra={"key": key})

    def _get_deployed_symlinks(self, key: str) -> list[Path]:
        """Get all deployed symlinks for a key (deployed/{key} or deployed/{key}-{md5})."""
        if not self.deployed_path.exists():
            return []
        symlinks = []
        for path in self.deployed_path.iterdir():
            if path.is_symlink():
                name = path.name
                # Exact match (old style: {key})
                if name == key:
                    symlinks.append(path)
                # New style: {key}-{md5} - use rsplit to avoid matching siblings
                elif "-" in name:
                    adapter_key, _ = name.rsplit("-", 1)
                    if adapter_key == key:
                        symlinks.append(path)
        return sorted(symlinks)

    def _migrate_old_symlink(self, key: str) -> None:
        """Migrate old-style deployed/{key} symlink to deployed/{key}-{md5}."""
        old_symlink = self.deployed_path / key
        if not old_symlink.is_symlink():
            return

        target = old_symlink.resolve()
        if not target.exists():
            # Remove dangling symlink instead of leaving it
            old_symlink.unlink()
            self._lg.warning("removed dangling symlink", extra={"key": key})
            return

        version_id = target.name
        config = self._read_adapter_config(key, version_id)
        if not config:
            self._lg.warning("migration skipped: no config", extra={"key": key})
            return

        md5 = config.get("md5", "unknown")
        new_symlink = self.deployed_path / f"{key}-{md5}"
        # Check both exists() and is_symlink() to handle dangling symlinks
        if not (new_symlink.exists() or new_symlink.is_symlink()):
            new_symlink.symlink_to(Path("..") / "adapters" / key / version_id)
            self._lg.info("migrated symlink", extra={"old": key, "new": new_symlink.name})

        old_symlink.unlink()

    def _resolve_deploy_version(self, key: str, version_id: str | None) -> tuple[str, str]:
        """Resolve version_id and md5 for deployment.

        Returns:
            Tuple of (version_id, md5).

        Raises:
            ValueError: If version not found.
        """
        if version_id is None:
            version_id = self._get_latest_version(key)
            if version_id is None:
                raise ValueError(f"No versions found for adapter '{key}'")

        version_path = self.adapters_path / key / version_id
        if not version_path.exists():
            raise ValueError(f"Version '{version_id}' not found for adapter '{key}'")

        config = self._read_adapter_config(key, version_id)
        md5 = config.get("md5", "unknown")
        return version_id, md5

    def deploy_adapter(
        self,
        key: str,
        version_id: str | None = None,
        *,
        policy: Literal["add", "replace"] = "replace",
    ) -> None:
        """Deploy an adapter version with versioned naming."""
        self.validate_key(key)
        self.deployed_path.mkdir(parents=True, exist_ok=True)
        self._migrate_old_symlink(key)

        version_id, md5 = self._resolve_deploy_version(key, version_id)

        # Remove existing symlinks if policy is "replace"
        if policy == "replace":
            for symlink in self._get_deployed_symlinks(key):
                symlink.unlink()
                self._lg.info("removed deployment", extra={"symlink": symlink.name})

        # Create new versioned symlink: deployed/{key}-{md5}
        symlink_path = self.deployed_path / f"{key}-{md5}"
        if symlink_path.exists() or symlink_path.is_symlink():
            symlink_path.unlink()

        relative_target = Path("..") / "adapters" / key / version_id
        symlink_path.symlink_to(relative_target)
        self._lg.info(
            "deployed adapter",
            extra={"key": key, "version": version_id, "md5": md5, "policy": policy},
        )

    def undeploy_adapter(self, key: str, md5: str | None = None) -> None:
        """Undeploy an adapter.

        Args:
            key: Adapter key.
            md5: Specific version to undeploy. If None, undeploy all versions.
        """
        self.validate_key(key)

        # Migrate any old-style symlinks first
        self._migrate_old_symlink(key)

        if md5 is not None:
            # Undeploy specific version
            symlink_path = self.deployed_path / f"{key}-{md5}"
            if symlink_path.is_symlink():
                symlink_path.unlink()
                self._lg.info("undeployed adapter", extra={"key": key, "md5": md5})
        else:
            # Undeploy all versions of this key
            for symlink in self._get_deployed_symlinks(key):
                symlink.unlink()
                self._lg.info("undeployed adapter", extra={"symlink": symlink.name})

    def is_deployed(self, key: str, md5: str | None = None) -> bool:
        """Check if adapter is deployed.

        Args:
            key: Adapter key.
            md5: Specific version to check. If None, check if any version is deployed.

        Returns:
            True if deployed.
        """
        self.validate_key(key)

        if md5 is not None:
            # Check specific version
            return (self.deployed_path / f"{key}-{md5}").is_symlink()

        # Check if any version is deployed (new or old style)
        return len(self._get_deployed_symlinks(key)) > 0

    def list_deployed(self, key: str | None = None) -> list[tuple[str, str]]:
        """List deployed adapters.

        Args:
            key: Filter to specific adapter key. If None, list all deployed.

        Returns:
            List of (key, md5) tuples for deployed adapters.
        """
        if not self.deployed_path.exists():
            return []

        result: list[tuple[str, str]] = []
        for symlink in sorted(self.deployed_path.iterdir()):
            if not symlink.is_symlink():
                continue
            name = symlink.name
            # Parse {key}-{md5} format
            # Note: This parsing assumes md5 never contains dashes (true for hex hashes
            # and the "unknown" fallback). Keys can contain dashes.
            if "-" in name:
                parts = name.rsplit("-", 1)
                if len(parts) == 2:
                    adapter_key, md5 = parts
                    # Normalize to lowercase for consistent comparison
                    md5_lower = md5.lower()
                    # Validate md5 is hex or "unknown" to avoid false matches
                    is_valid_md5 = md5_lower == "unknown" or (
                        len(md5_lower) <= 32 and all(c in "0123456789abcdef" for c in md5_lower)
                    )
                    if is_valid_md5 and (key is None or adapter_key == key):
                        result.append((adapter_key, md5_lower))
            else:
                # Old-style symlink without md5 - treat as unknown
                if key is None or name == key:
                    result.append((name, "unknown"))
        return result

    def get_deployed_path(self, key: str) -> Path | None:
        """Get filesystem path to deployed adapter.

        If multiple versions are deployed, returns the most recently modified.
        """
        self.validate_key(key)

        # Get all deployed symlinks for this key
        symlinks = self._get_deployed_symlinks(key)
        if not symlinks:
            return None

        # If only one, return it
        if len(symlinks) == 1:
            return symlinks[0].resolve()

        # Multiple deployed - return the one with latest mtime
        latest = max(symlinks, key=lambda p: p.lstat().st_mtime)
        return latest.resolve()

    # =========================================================================
    # Work Area Operations (ABC implementation)
    # =========================================================================

    def create_work_area(self, adapter: str, clean: bool = True) -> Path:
        """Create a temporary work area for training."""
        self.validate_key(adapter)
        work_dir = self.work_path / adapter

        if clean and work_dir.exists():
            self._lg.info("cleaning work area", extra={"adapter": adapter})
            shutil.rmtree(work_dir)

        work_dir.mkdir(parents=True, exist_ok=True)
        return work_dir

    def cleanup_work_area(self, adapter: str) -> None:
        """Clean up a training work area."""
        self.validate_key(adapter)
        work_dir = self.work_path / adapter
        if work_dir.exists():
            shutil.rmtree(work_dir)

    def write_training_data(self, work_dir: Path, adapter: str, records: list[dict]) -> Path:
        """Write training data to work area."""
        work_dir.mkdir(parents=True, exist_ok=True)
        data_path = work_dir / f"{adapter}.jsonl"

        with data_path.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        return data_path

    def resolve_data_path(self, manifest: Manifest, work_dir: Path) -> Path:
        """Resolve manifest data to a file path."""
        if manifest.data.format == "inline":
            return self.write_training_data(work_dir, manifest.adapter, manifest.data.records)

        # External file - resolve path
        if not manifest.data.path:
            raise ValueError("External data format requires a path")
        data_path = Path(manifest.data.path)
        if not data_path.is_absolute():
            # Relative path - resolve against manifest location or work_dir
            if manifest.source_path:
                data_path = manifest.source_path.parent / data_path
            else:
                data_path = work_dir / data_path

        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        return data_path

    # =========================================================================
    # Validation (ABC implementation)
    # =========================================================================

    def validate_key(self, key: str) -> None:
        """Validate an adapter/manifest key.

        Keys must be alphanumeric with hyphens, underscores, or dots.
        No path traversal, whitespace, or special characters allowed.
        """
        if not key:
            raise ValueError("Key cannot be empty")
        if "/" in key or "\\" in key or ".." in key:
            raise ValueError(f"Invalid key (path traversal): {key}")
        # Allow alphanumeric, hyphen, underscore, dot (but not leading/trailing dots)
        if not _KEY_RE.match(key):
            raise ValueError(
                f"Invalid key '{key}': must be alphanumeric with hyphens, underscores, or dots"
            )

    # =========================================================================
    # Filesystem Helpers (not in ABC - for backwards compatibility)
    # =========================================================================

    def ensure_directories(self) -> None:
        """Create all required registry directories."""
        for path in (
            self.pending_path,
            self.completed_path,
            self.adapters_path,
            self.deployed_path,
            self.work_path,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def ensure_work_dir(self, adapter: str, *, clean: bool = True) -> Path:
        """Create work directory for training.

        Alias for create_work_area for backwards compatibility.
        """
        return self.create_work_area(adapter, clean=clean)

    def clean_work_dir(self, adapter: str) -> None:
        """Remove work directory for an adapter.

        Alias for cleanup_work_area for backwards compatibility.
        """
        self.cleanup_work_area(adapter)

    def ensure_dir(self, path: Path) -> Path:
        """Ensure a directory exists.

        Generic helper for creating any directory (including user-provided paths).

        Args:
            path: Directory path to create.

        Returns:
            The path (for chaining).
        """
        path.mkdir(parents=True, exist_ok=True)
        return path

    def delete_file(self, path: Path) -> None:
        """Delete a file if it exists."""
        if path.exists():
            path.unlink()

    # =========================================================================
    # Internal Helpers
    # =========================================================================

    def _get_latest_version(self, key: str) -> str | None:
        """Get latest version ID for an adapter."""
        key_path = self.adapters_path / key
        if not key_path.is_dir():
            return None

        versions = sorted(p.name for p in key_path.iterdir() if p.is_dir())
        return versions[-1] if versions else None

    def _read_adapter_config(self, key: str, version_id: str) -> dict[str, Any]:
        """Read config.yaml for an adapter version."""
        config_path = self.adapters_path / key / version_id / "config.yaml"
        if not config_path.exists():
            return {}
        with config_path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    # =========================================================================
    # Legacy Methods (for backwards compatibility during migration)
    # =========================================================================

    def write_manifest(self, path: Path, data: dict[str, Any], *, compress: bool = False) -> Path:
        """Write manifest YAML to file (legacy - accepts dict)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        yaml_content = yaml.safe_dump(data, sort_keys=False, allow_unicode=True)

        if compress:
            if path.suffix != ".gz":
                path = path.with_suffix(path.suffix + ".gz")
            with gzip.open(path, "wt", encoding="utf-8") as f:
                f.write(yaml_content)
        else:
            with path.open("w", encoding="utf-8") as f:
                f.write(yaml_content)

        return path

    def remove_pending(self, adapter: str) -> None:
        """Remove manifest from pending queue (legacy)."""
        path = self.get_pending_path(adapter)
        if not path.exists():
            raise FileNotFoundError(f"Manifest not in queue: {adapter}")
        path.unlink()

    def move_to_completed(
        self, manifest_path: Path, adapter: str, md5: str, *, compress: bool = True
    ) -> Path:
        """Move manifest to completed directory (legacy)."""
        self.completed_path.mkdir(parents=True, exist_ok=True)

        suffix = ".yaml.gz" if compress else ".yaml"
        completed_path = self.completed_path / f"{adapter}-{md5}{suffix}"

        # Read, write to new location, delete original
        data = self.read_manifest(manifest_path)
        self.write_manifest(completed_path, data, compress=compress)

        if manifest_path.exists():
            manifest_path.unlink()

        return completed_path

    def resolve_external_data(
        self, data_path: str | Path, manifest_dir: Path | None = None
    ) -> Path:
        """Resolve external data file path (legacy)."""
        path = Path(data_path)
        if path.is_absolute():
            resolved = path
        elif manifest_dir:
            resolved = manifest_dir / path
        else:
            resolved = path

        if not resolved.exists():
            raise FileNotFoundError(f"Data file not found: {resolved}")

        return resolved

    def get_adapter_path(self, key: str, version_id: str) -> Path:
        """Get path to specific adapter version (legacy)."""
        self.validate_key(key)
        return self.adapters_path / key / version_id

    def version_exists(self, key: str, version_id: str) -> bool:
        """Check if adapter version exists."""
        return self.get_adapter_path(key, version_id).exists()

    def list_pending(self) -> list[Path]:
        """List pending manifest files (legacy - returns paths)."""
        if not self.pending_path.exists():
            return []
        return sorted(self.pending_path.glob("*.yaml"))

    def list_completed(self) -> list[Path]:
        """List completed manifest files (legacy - returns paths)."""
        if not self.completed_path.exists():
            return []
        yaml_files = list(self.completed_path.glob("*.yaml"))
        gz_files = list(self.completed_path.glob("*.yaml.gz"))
        return sorted(yaml_files + gz_files)

    def read_manifest(self, path: Path) -> dict[str, Any]:
        """Load manifest YAML from file (legacy - returns dict)."""
        if not path.exists():
            raise FileNotFoundError(f"Manifest not found: {path}")

        if path.suffix == ".gz":
            with gzip.open(path, "rt", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def get_pending_path(self, adapter: str) -> Path:
        """Get path to pending manifest for an adapter."""
        self.validate_key(adapter)
        return self.pending_path / f"{adapter}.yaml"

    def pending_exists(self, adapter: str) -> bool:
        """Check if pending manifest exists."""
        return self.get_pending_path(adapter).exists()

    def adapter_exists(self, key: str) -> bool:
        """Check if adapter is registered."""
        self.validate_key(key)
        return (self.adapters_path / key).exists()

    def list_versions(self, key: str) -> list[str]:
        """List versions for an adapter."""
        self.validate_key(key)
        key_path = self.adapters_path / key
        if not key_path.is_dir():
            return []
        return sorted(p.name for p in key_path.iterdir() if p.is_dir())

    def get_latest_version(self, key: str) -> str | None:
        """Get latest version ID for an adapter (public API)."""
        return self._get_latest_version(key)

    def get_latest_version_path(self, key: str) -> Path | None:
        """Get path to latest version of an adapter."""
        version = self._get_latest_version(key)
        if version is None:
            return None
        return self.adapters_path / key / version

    def get_deployed_version(self, key: str) -> str | None:
        """Get deployed version ID for an adapter.

        If multiple versions are deployed, returns the most recently modified.
        """
        path = self.get_deployed_path(key)
        if path is None:
            return None
        return path.name

    def get_deployed_version_path(self, key: str) -> Path | None:
        """Get path to deployed version of an adapter."""
        return self.get_deployed_path(key)

    def store_adapter_files(self, source: Path, key: str, version_id: str) -> Path:
        """Copy adapter files to registry (low-level).

        This is used by AdapterRegistry for more control over the storage process.
        """
        self.validate_key(key)
        if not source.exists():
            raise FileNotFoundError(f"Adapter source not found: {source}")

        adapter_path = self.adapters_path / key / version_id
        if adapter_path.exists():
            shutil.rmtree(adapter_path)

        shutil.copytree(source, adapter_path)
        self._lg.info("stored adapter files", extra={"key": key, "version": version_id})

        return adapter_path

    def read_adapter_config(self, key: str, version_id: str) -> dict[str, Any]:
        """Read config.yaml for an adapter version (public API)."""
        return self._read_adapter_config(key, version_id)

    def write_adapter_config(self, key: str, version_id: str, config: dict[str, Any]) -> Path:
        """Write config.yaml for an adapter version."""
        config_path = self.adapters_path / key / version_id / "config.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with config_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(config, f)
        self._lg.info("wrote adapter config", extra={"key": key, "version": version_id})
        return config_path

    def deploy(
        self,
        key: str,
        version_id: str | None = None,
        *,
        policy: Literal["add", "replace"] = "replace",
    ) -> None:
        """Create deployment symlink (alias for deploy_adapter)."""
        self.deploy_adapter(key, version_id, policy=policy)

    def undeploy(self, key: str, md5: str | None = None) -> None:
        """Remove deployment symlink (alias for undeploy_adapter)."""
        self.undeploy_adapter(key, md5)

    def write_data_file(self, work_dir: Path, filename: str, records: list[dict[str, Any]]) -> Path:
        """Write training data to JSONL file (legacy API with explicit filename)."""
        work_dir.mkdir(parents=True, exist_ok=True)
        data_path = work_dir / filename

        with data_path.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        return data_path

    def iter_adapter_keys(self) -> list[tuple[str, Path]]:
        """Iterate over adapter keys with their paths."""
        if not self.adapters_path.exists():
            return []
        return [(p.name, p) for p in self.adapters_path.iterdir() if p.is_dir()]
