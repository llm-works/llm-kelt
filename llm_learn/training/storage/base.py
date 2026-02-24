"""Abstract base class for training storage.

Defines the interface for storing and retrieving training artifacts:
- Manifests (pending and completed)
- Adapters (with versioning and deployment)
- Training work areas (temporary, for HuggingFace)

Implementations can be file-based (FileStorage) or database-backed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from ..manifest.schema import Manifest
    from ..schema import AdapterInfo, RunResult


class DupAdapterError(ValueError):
    """Raised when attempting to store an adapter that already exists with the same md5."""

    def __init__(self, key: str, md5: str):
        self.key = key
        self.md5 = md5
        super().__init__(
            f"Adapter '{key}' already exists with md5 {md5[:8]}. "
            "Use overwrite=True to return existing."
        )


class Storage(ABC):
    """Abstract storage interface for training artifacts.

    This interface abstracts away the storage mechanism, allowing
    implementations to be file-based, database-backed, or cloud-based.

    Training work areas must provide real filesystem paths because
    HuggingFace/TRL libraries require disk access. Everything else
    can be stored abstractly.
    """

    # =========================================================================
    # Manifest Operations
    # =========================================================================

    @abstractmethod
    def submit_manifest(self, manifest: Manifest) -> None:
        """Submit a manifest to the pending queue.

        Args:
            manifest: Manifest to submit.

        Raises:
            ValueError: If manifest with same adapter key already pending.
        """

    @abstractmethod
    def get_pending_manifest(self, adapter: str) -> Manifest | None:
        """Get a pending manifest by adapter key.

        Args:
            adapter: Adapter key.

        Returns:
            Manifest or None if not found.
        """

    @abstractmethod
    def list_pending_manifests(self) -> list[Manifest]:
        """List all pending manifests.

        Returns:
            List of pending manifests.
        """

    @abstractmethod
    def remove_pending_manifest(self, adapter: str) -> None:
        """Remove a manifest from pending queue.

        Args:
            adapter: Adapter key.

        Raises:
            FileNotFoundError: If manifest not found.
        """

    @abstractmethod
    def complete_manifest(self, manifest: Manifest) -> None:
        """Move a manifest to completed storage.

        The manifest should have output populated with training results.

        Args:
            manifest: Completed manifest with output.
        """

    @abstractmethod
    def list_completed_manifests(self) -> list[Manifest]:
        """List all completed manifests.

        Returns:
            List of completed manifests.
        """

    @abstractmethod
    def find_adapter_by_md5(self, md5: str) -> Manifest | None:
        """Find a completed manifest by adapter MD5.

        Args:
            md5: MD5 hash of adapter weights.

        Returns:
            Manifest if found, None otherwise.
        """

    # =========================================================================
    # Adapter Operations
    # =========================================================================

    @abstractmethod
    def store_adapter(
        self,
        training_result: RunResult,
        key: str,
        description: str,
        deploy: bool | Literal["add", "replace"] = True,
    ) -> AdapterInfo:
        """Store a trained adapter.

        Args:
            training_result: Result from training with adapter path.
            key: Adapter key.
            description: Human-readable description.
            deploy: Deployment setting:
                - True or "replace": Deploy and remove existing {key}-* symlinks.
                - "add": Deploy and keep existing {key}-* symlinks.
                - False: Don't deploy.

        Returns:
            AdapterInfo with registration details.

        Raises:
            ValueError: If adapter already exists.
        """

    @abstractmethod
    def get_adapter(self, key: str) -> AdapterInfo | None:
        """Get adapter info by key.

        Args:
            key: Adapter key.

        Returns:
            AdapterInfo or None if not found.
        """

    @abstractmethod
    def get_adapter_by_md5(self, key: str, md5: str) -> AdapterInfo | None:
        """Get adapter info by key and md5.

        Args:
            key: Adapter key.
            md5: MD5 hash to match.

        Returns:
            AdapterInfo or None if not found.
        """

    @abstractmethod
    def list_adapters(self) -> list[str]:
        """List all registered adapter keys.

        Returns:
            List of adapter keys.
        """

    @abstractmethod
    def list_adapter_infos(self) -> list[AdapterInfo]:
        """List all registered adapters with full info.

        Returns:
            List of adapter info objects.
        """

    @abstractmethod
    def remove_adapter(self, key: str, version_id: str | None = None) -> None:
        """Remove an adapter or specific version.

        Args:
            key: Adapter key.
            version_id: Specific version to remove (all versions if None).

        Raises:
            ValueError: If adapter not found.
        """

    @abstractmethod
    def deploy_adapter(
        self,
        key: str,
        version_id: str | None = None,
        *,
        policy: Literal["add", "replace"] = "replace",
    ) -> None:
        """Deploy an adapter version.

        Creates a symlink in deployed/ directory with versioned naming: {key}-{md5}.
        This allows multiple versions of the same adapter to be deployed simultaneously.

        Args:
            key: Adapter key.
            version_id: Version to deploy (latest if None).
            policy: Deployment policy:
                - "add": Create symlink, keep existing {key}-* symlinks.
                - "replace": Create symlink, remove existing {key}-* symlinks.

        Raises:
            ValueError: If adapter not found.
        """

    @abstractmethod
    def undeploy_adapter(self, key: str, md5: str | None = None) -> None:
        """Undeploy an adapter.

        Args:
            key: Adapter key.
            md5: Specific version to undeploy. If None, undeploy all versions of key.
        """

    @abstractmethod
    def is_deployed(self, key: str, md5: str | None = None) -> bool:
        """Check if adapter is deployed.

        Args:
            key: Adapter key.
            md5: Specific version to check. If None, check if any version is deployed.

        Returns:
            True if deployed (any version if md5 is None, specific version otherwise).
        """

    @abstractmethod
    def list_deployed(self, key: str | None = None) -> list[tuple[str, str]]:
        """List deployed adapters.

        Args:
            key: Filter to specific adapter key. If None, list all deployed.

        Returns:
            List of (key, md5) tuples for deployed adapters.
        """

    @abstractmethod
    def get_deployed_path(self, key: str) -> Path | None:
        """Get filesystem path to deployed adapter.

        This is needed for inference servers that read from disk.

        Args:
            key: Adapter key.

        Returns:
            Path to deployed adapter, or None if not deployed.
        """

    # =========================================================================
    # Work Area Operations (filesystem-based for HuggingFace)
    # =========================================================================

    @abstractmethod
    def create_work_area(self, adapter: str, clean: bool = True) -> Path:
        """Create a temporary work area for training.

        HuggingFace/TRL requires real filesystem paths for training.
        This creates a temporary directory that will be cleaned up
        after training completes.

        Args:
            adapter: Adapter key (used as directory name).
            clean: If True, remove existing work area first.

        Returns:
            Path to work directory.
        """

    @abstractmethod
    def cleanup_work_area(self, adapter: str) -> None:
        """Clean up a training work area.

        Args:
            adapter: Adapter key.
        """

    @abstractmethod
    def write_training_data(self, work_dir: Path, adapter: str, records: list[dict]) -> Path:
        """Write training data to work area.

        Args:
            work_dir: Work directory path.
            adapter: Adapter key (used in filename).
            records: Training records to write.

        Returns:
            Path to written data file.
        """

    @abstractmethod
    def resolve_data_path(self, manifest: Manifest, work_dir: Path) -> Path:
        """Resolve manifest data to a file path.

        For inline data, writes to work_dir and returns path.
        For external data, resolves the path:
        - Absolute paths used as-is
        - Relative paths resolved against manifest.source_path (if set) or work_dir

        Args:
            manifest: Manifest with data section.
            work_dir: Work directory for inline data.

        Returns:
            Path to data file.
        """

    # =========================================================================
    # Validation
    # =========================================================================

    @abstractmethod
    def validate_key(self, key: str) -> None:
        """Validate an adapter/manifest key.

        Args:
            key: Key to validate.

        Raises:
            ValueError: If key is invalid.
        """
