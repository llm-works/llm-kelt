# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-kelt Authors

"""File-based storage implementation.

Implements the Storage interface using the local filesystem.
"""

from __future__ import annotations

import fcntl
import gzip
import json
import os
import re
import shutil
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import yaml
from appinfra.cas import FileStore
from appinfra.log import Logger

from ..lineage import DEFAULT_SERIES, Lineage, series_root
from ..manifest.schema import Data, Manifest
from .base import SERIES_SEPARATOR, Storage, queue_key

# Pre-compiled regex for key validation (used on hot path)
_KEY_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]*[a-zA-Z0-9]$|^[a-zA-Z0-9]$")

if TYPE_CHECKING:
    from ..schema import ActiveSeries, AdapterInfo, QueueStatus, RunResult, SubmitResult


def _series_root(manifest: Manifest) -> dict[str, Any]:
    """Return the base model and recipe a series manifest trains with."""
    training = {
        k: v for k, v in manifest.training.items() if k not in ("requested_model", "base_model")
    }
    recipe = {
        "method": manifest.method,
        "lora": dict(manifest.lora),
        "training": training,
        "method_config": dict(manifest.method_config),
    }
    return series_root(str(manifest.training["requested_model"]), recipe)


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
        ├── running/       # Manifests claimed by a runner, training in progress
        ├── completed/     # Finished manifests (archive)
        ├── failed/        # Manifests whose training failed, with the error in output
        ├── adapters/      # All adapters stored here
        │   └── {key}/{version_id}/
        ├── deployed/      # Symlinks to enabled adapters
        ├── work/          # Temporary training directories
        ├── cas/           # Training sets of series, one repository per adapter key
        └── lineage/       # Submit dates and builds of series commits
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
        self.running_path = self.base_path / "running"
        self.completed_path = self.base_path / "completed"
        self.failed_path = self.base_path / "failed"
        self.adapters_path = self.base_path / "adapters"
        self.deployed_path = self.base_path / "deployed"
        self.work_path = self.base_path / "work"
        self._lineage = Lineage(
            lg, FileStore(lg, self.base_path / "cas"), self.base_path / "lineage"
        )

    @property
    def lineage(self) -> Lineage:
        """Series of training sets per adapter key."""
        return self._lineage

    # =========================================================================
    # Manifest Operations (ABC implementation)
    # =========================================================================

    def submit_manifest(self, manifest: Manifest) -> SubmitResult:
        """Submit a manifest to the pending queue.

        The manifest is written to a hidden temporary file and hard-linked into place, so
        readers of pending/ never see a partial file and a concurrent submit for the same
        key fails instead of overwriting.
        """
        from ..schema import SubmitResult

        if manifest.series:
            return self._submit_series(manifest, manifest.series)

        self.validate_key(manifest.adapter)
        pending_file = self.pending_path / f"{manifest.adapter}.yaml"
        if pending_file.exists() or not self._link_pending(manifest, pending_file):
            raise ValueError(f"Manifest already pending for adapter: {manifest.adapter}")
        self._lg.info("submitted manifest", extra={"adapter": manifest.adapter})

        return SubmitResult(
            adapter=manifest.adapter,
            timestamp=datetime.now(),
            location=str(pending_file),
        )

    def _link_pending(self, manifest: Manifest, pending_file: Path) -> bool:
        """Write a manifest to pending_file unless it exists; False if it does."""
        from ..manifest.loader import save_manifest

        self.pending_path.mkdir(parents=True, exist_ok=True)
        tmp_file = self.pending_path / f".{manifest.adapter}.{uuid.uuid4().hex}.tmp"
        try:
            save_manifest(manifest, tmp_file)
            os.link(tmp_file, pending_file)
        except FileExistsError:
            return False
        finally:
            tmp_file.unlink(missing_ok=True)
        return True

    def get_pending_manifest(self, adapter: str) -> Manifest | None:
        """Get a pending manifest by queue key."""
        from ..manifest.loader import load_manifest

        self.validate_queue_key(adapter)
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

    def list_pending_keys(self) -> list[str]:
        """List adapter keys of pending manifests, oldest submission first."""
        if not self.pending_path.exists():
            return []
        stamped = []
        for path in self.pending_path.glob("*.yaml"):
            try:
                stamped.append((path.stat().st_mtime, path.stem))
            except FileNotFoundError:
                continue
        return [key for _, key in sorted(stamped)]

    def remove_pending_manifest(self, adapter: str) -> None:
        """Remove a manifest from pending queue."""
        self.validate_queue_key(adapter)
        pending_file = self.pending_path / f"{adapter}.yaml"
        if not pending_file.exists():
            raise FileNotFoundError(f"Manifest not found: {adapter}")
        pending_file.unlink()

    def claim_pending_manifest(self, adapter: str) -> Manifest:
        """Move a pending manifest to running/ and return it.

        The rename is atomic, so two runners cannot claim the same manifest. A relative
        external data path is anchored to pending/, where the producer wrote it.

        Raises:
            FileNotFoundError: No pending manifest for the key (or another runner took it).
            FileExistsError: A manifest for the key is already running.
            ValueError, yaml.YAMLError: The claimed manifest cannot be parsed; it stays in
                running/.
        """
        from ..manifest.loader import load_manifest

        self.validate_queue_key(adapter)
        self.running_path.mkdir(parents=True, exist_ok=True)
        running_file = self.running_path / f"{adapter}.yaml"
        pending_file = self.pending_path / f"{adapter}.yaml"
        try:
            os.link(pending_file, running_file)
        except FileExistsError:
            raise FileExistsError(f"Manifest already running for adapter: {adapter}") from None
        pending_file.unlink()
        self._lg.info("claimed manifest", extra={"adapter": adapter})

        manifest = load_manifest(running_file)
        if manifest.series:
            self._resolve_series_build(manifest, manifest.series, running_file)
            return manifest
        data_path = manifest.data.get("path")
        if manifest.data.format == "external" and data_path and not Path(data_path).is_absolute():
            manifest.data.path = str(self.pending_path / data_path)
        return manifest

    def fail_running_manifest(self, adapter: str, result: RunResult) -> Path:
        """Move a running manifest to failed/, recording the failed result as its output.

        A manifest that cannot be parsed is moved unchanged.
        """
        from ..manifest.loader import load_manifest, save_manifest

        self.validate_queue_key(adapter)
        self.failed_path.mkdir(parents=True, exist_ok=True)
        running_file = self.running_path / f"{adapter}.yaml"
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")

        try:
            manifest = load_manifest(running_file)
        except (yaml.YAMLError, ValueError, TypeError) as e:
            self._lg.warning(
                "failed manifest is unreadable, moving it unchanged",
                extra={"adapter": adapter, "exception": e},
            )
            failed_file = self.failed_path / f"{adapter}-{stamp}.yaml"
            running_file.rename(failed_file)
            return failed_file

        if manifest.commit:
            self._lineage.record_build(
                manifest.adapter, manifest.commit, "failed", error=result.error
            )
        manifest.output = result
        failed_file = self.failed_path / f"{adapter}-{stamp}.yaml.gz"
        save_manifest(manifest, failed_file, compress=True)
        running_file.unlink()
        self._lg.info("failed manifest", extra={"adapter": adapter, "error": result.error})
        return failed_file

    # =========================================================================
    # Series (see training/lineage.py)
    # =========================================================================

    def _submit_series(self, manifest: Manifest, series: str) -> SubmitResult:
        """Add a series manifest's data to its series and queue a build of the tip.

        The queued manifest carries no data: the runner builds whatever the series' tip
        is when it claims the build, so a submit while a build is pending queues nothing.
        """
        from ..schema import SubmitResult

        self._check_series_manifest(manifest, series)
        submission = self._lineage.submit(
            manifest.adapter,
            series,
            self._read_records(manifest),
            dict(manifest.data.get("templates") or {}),
            _series_root(manifest),
            dict(manifest.get("annotations") or {}),
        )
        pending_file = self.pending_path / f"{queue_key(manifest.adapter, series)}.yaml"
        if self._needs_build(manifest.adapter, series):
            # The queued build is of the series' tip, not of this set
            empty: dict[str, Any] = {
                "data": Data(format="inline"),
                "commit": None,
                "annotations": {},
            }
            self._link_pending(Manifest({**manifest, **empty}), pending_file)  # False: queued
        return SubmitResult(
            adapter=manifest.adapter,
            timestamp=datetime.now(),
            location=str(pending_file),
            commit=submission.commit,
            created=submission.created,
            root=submission.root,
        )

    def _check_series_manifest(self, manifest: Manifest, series: str) -> None:
        """Raise ValueError if a series manifest cannot be submitted."""
        self.validate_key(manifest.adapter)
        self.validate_key(series)
        if not manifest.training.get("requested_model"):
            raise ValueError("A series manifest requires a base model")
        if manifest.get("parent") is not None:
            raise ValueError("A series manifest cannot have a parent adapter")
        policy = manifest.deployment.policy if manifest.deployment else "replace"
        if series != DEFAULT_SERIES and policy != "skip":
            raise ValueError(
                f"Only series '{DEFAULT_SERIES}' is deployed; "
                f"use deployment policy 'skip' for series '{series}'"
            )
        # Series "main" shares its queue key with one-off manifests of the adapter
        queued = self._queued_metadata(self.pending_path, queue_key(manifest.adapter, series))
        if queued is not None and queued.series != series:
            raise ValueError(f"Manifest already pending for adapter: {manifest.adapter}")

    def _queued_metadata(self, queue_dir: Path, key: str) -> Manifest | None:
        """Return the metadata of the manifest queued under a key in a queue dir, or None."""
        from ..manifest.loader import load_manifest_metadata

        try:
            return load_manifest_metadata(queue_dir / f"{key}.yaml")
        except FileNotFoundError:
            return None

    def _read_records(self, manifest: Manifest) -> list[dict[str, Any]]:
        """Return a manifest's records, inline or from its external JSONL file."""
        data = manifest.data
        if data.format == "inline":
            records = list(data.records)
        else:
            if not data.path:
                raise ValueError("External data requires a path")
            path = Path(data.path)
            if not path.is_absolute() and manifest.source_path is not None:
                path = manifest.source_path.parent / path
            lines = path.read_text(encoding="utf-8").splitlines()
            records = [json.loads(line) for line in lines if line.strip()]
        if not records:
            raise ValueError("A series manifest requires at least one record")
        return records

    def _needs_build(self, adapter: str, series: str) -> bool:
        """True unless the series' tip is built or being built."""
        tip = self._lineage.tip(adapter, series)
        build = self._lineage.build(adapter, tip) if tip else None
        if build is not None and build.state == "completed":
            return False
        key = queue_key(adapter, series)
        if self._queued_metadata(self.pending_path, key) is not None:
            return False  # Already queued
        running = self._queued_metadata(self.running_path, key)
        return running is None or running.commit != tip

    def _resolve_series_build(self, manifest: Manifest, series: str, running_file: Path) -> None:
        """Pin a claimed series build to the series' tip and fill in its training data.

        The running file records the commit, so a failed build is recorded against it.
        """
        from ..manifest.loader import save_manifest

        commit = self._lineage.tip(manifest.adapter, series)
        if commit is None:
            raise ValueError(f"Series not found: {manifest.adapter}/{series}")
        manifest.commit = commit
        save_manifest(manifest, running_file)
        records, templates, root = self._lineage.build_input(manifest.adapter, commit)
        manifest.training["requested_model"] = root["base"]["model"]
        manifest.data = Data(format="inline", records=records, templates=templates)

    # =========================================================================
    # Series deployment and retention
    # =========================================================================

    def deploy_series(
        self,
        adapter: str,
        series: str,
        commit: str | None = None,
        *,
        policy: Literal["add", "replace"] = "replace",
    ) -> str:
        """Deploy the adapter version built from a commit of a series; return its version id.

        Raises:
            ValueError: The series or commit does not exist, or the commit has no completed
                build (never built, failed, or its weights were pruned).
        """
        self.validate_key(adapter)
        commit, version_id = self._series_build_version(adapter, series, commit)
        # Recorded first: until the link is in place, the record does not match it
        self._lineage.record_deployed(adapter, series, commit)
        self.deploy_adapter(adapter, version_id, policy=policy)
        return version_id

    def _series_build_version(
        self, adapter: str, series: str, commit: str | None
    ) -> tuple[str, str]:
        """Return (commit, version id) of a completed series build; default: current build."""
        info = self._lineage.series(adapter, series)
        if info is None:
            raise ValueError(f"Series not found: {adapter}/{series}")
        if commit is None:
            entry = info.current_build
            if entry is None:
                raise ValueError(f"Series {adapter}/{series} has no completed build")
        else:
            entry = next((e for e in info.sets if e.commit == commit), None)
            if entry is None:
                raise ValueError(f"Commit {commit} is not in series {adapter}/{series}")
        build = entry.build
        if build is None or build.state != "completed" or not build.md5:
            state = build.state if build else "not built"
            raise ValueError(f"Commit {entry.commit} of {adapter}/{series} is {state}")
        version_id = self._find_version_by_md5(adapter, build.md5)
        if version_id is None:
            raise ValueError(f"No adapter version with md5 {build.md5} for {adapter}")
        return entry.commit, version_id

    def active_series(self, adapter: str) -> ActiveSeries | None:
        """Return the series the adapter key serves, or None if no series build is deployed.

        With several versions deployed, the most recently deployed one counts.
        """
        from ..schema import ActiveSeries

        path = self.get_deployed_path(adapter)
        if path is None:
            return None
        config = self._read_adapter_config(adapter, path.name)
        commit, series = config.get("commit"), config.get("series")
        if not commit or not series:
            return None  # a one-off training
        recorded = self._lineage.deployed(adapter)
        if recorded is not None and recorded[1] == commit:
            series = recorded[0]
        return ActiveSeries(
            adapter=adapter,
            series=series,
            commit=commit,
            version_id=path.name,
            md5=config.get("md5", "unknown"),
        )

    def prune_series_adapters(self, adapter: str, series: str) -> list[str]:
        """Delete the adapter versions of a series' builds, except deployed ones.

        The sets and build records stay; each build is marked pruned, with the md5 of its
        weights, and can be rebuilt. Builds a series shares with a fork are pruned for both.

        Returns:
            The commits whose builds were pruned, oldest first.
        """
        info = self._lineage.series(adapter, series)
        if info is None:
            raise ValueError(f"Series not found: {adapter}/{series}")
        pruned = []
        for entry in info.sets:
            build = entry.build
            if build is None or build.state != "completed" or not build.md5:
                continue
            with self._deploy_lock(adapter):
                if self.is_deployed(adapter, build.md5):
                    continue
                self._remove_build_weights(adapter, build.md5)
            self._lineage.mark_pruned(adapter, entry.commit)
            pruned.append(entry.commit)
        self._lg.info("pruned series adapters", extra={"adapter": adapter, "series": series})
        return pruned

    def drop_series(self, adapter: str, series: str) -> list[str]:
        """Delete a series with the sets, builds and adapter versions only it holds.

        Sets and builds a series shares with another series stay. A queued build of the
        series is removed. Returns the dropped commits, oldest first.

        Raises:
            ValueError: The series is active or has a deployed build, or a build of it is
                running.
        """
        self.validate_key(adapter)
        self.validate_key(series)
        active = self.active_series(adapter)
        if active is not None and active.series == series:
            raise ValueError(f"Series {adapter}/{series} is deployed; deploy another first")
        key = queue_key(adapter, series)
        if self._queued_metadata(self.running_path, key) is not None:
            raise ValueError(f"A build of {adapter}/{series} is running")
        exclusive = self._lineage.exclusive_commits(adapter, series)
        builds = [self._lineage.build(adapter, c) for c in exclusive]
        md5s = [b.md5 for b in builds if b is not None and b.md5]

        queued = self._queued_metadata(self.pending_path, key)
        if queued is not None and queued.series == series:
            (self.pending_path / f"{key}.yaml").unlink(missing_ok=True)
        # Lock: check deployed + remove weights atomically, so a concurrent deploy cannot
        # create a symlink to weights we are about to delete.
        with self._deploy_lock(adapter):
            if any(self.is_deployed(adapter, md5) for md5 in md5s):
                raise ValueError(f"A build of {adapter}/{series} is deployed; undeploy it first")
            # Weights go last: if the lineage refuses (the series moved), its builds keep them
            dropped = self._lineage.drop_series(adapter, series)
            for md5 in md5s:
                self._remove_build_weights(adapter, md5)
        return dropped

    def wipe_adapter(self, adapter: str) -> None:
        """Delete everything of an adapter key: series, sets, builds, versions, deployments.

        Queued manifests of the key are removed; completed and failed manifests stay as
        the run archive.

        Raises:
            ValueError: A build of the key is running.
        """
        self.validate_key(adapter)
        if self._queue_files(self.running_path, adapter):
            raise ValueError(f"A build of {adapter} is running")
        for path in self._queue_files(self.pending_path, adapter):
            path.unlink(missing_ok=True)
        with self._deploy_lock(adapter):
            if (self.adapters_path / adapter).exists():
                self.remove_adapter(adapter)
            self._lineage.wipe(adapter)

    def _queue_files(self, queue_dir: Path, adapter: str) -> list[Path]:
        """Queue files of an adapter key: its own and those of its series."""
        own = queue_dir / f"{adapter}.yaml"
        series = queue_dir.glob(f"{adapter}{SERIES_SEPARATOR}*.yaml")
        return [own, *series] if own.exists() else list(series)

    def _remove_build_weights(self, adapter: str, md5: str) -> None:
        """Remove the adapter version with these weights, if the registry still has it.

        Caller must hold _deploy_lock(adapter).
        """
        version_id = self._find_version_by_md5(adapter, md5)
        if version_id is not None:
            self._remove_version_unlocked(adapter, version_id)

    @contextmanager
    def _deploy_lock(self, adapter: str) -> Iterator[None]:
        """Hold the adapter's deploy lock, so deploy/remove operations do not interleave."""
        locks_dir = self.base_path / ".locks"
        locks_dir.mkdir(parents=True, exist_ok=True)
        path = locks_dir / f"{adapter}.deploy"
        with path.open("a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            yield

    def get_queue_status(self, adapter: str) -> QueueStatus | None:
        """Return the state of the latest submission for a queue key, or None.

        Reads manifest metadata only (not training data). A submission still queued wins
        over finished ones; among finished ones the most recently written file wins.
        """
        self.validate_queue_key(adapter)
        queued = [
            ("pending", self.pending_path / f"{adapter}.yaml"),
            ("running", self.running_path / f"{adapter}.yaml"),
        ]
        for state, path in queued + self._finished_by_recency(adapter):
            status = self._read_queue_status(adapter, state, path)
            if status is not None:
                return status
        return None

    def _finished_by_recency(self, adapter: str) -> list[tuple[str, Path]]:
        """Completed and failed manifest files whose name starts with the key, newest first."""
        candidates = [("completed", p) for p in self.completed_path.glob(f"{adapter}-*.yaml*")]
        candidates += [("failed", p) for p in self.failed_path.glob(f"{adapter}-*.yaml*")]
        mtimes = {}
        for state, path in candidates:
            try:
                mtimes[path] = path.stat().st_mtime
            except FileNotFoundError:
                continue
        found = [(state, path) for state, path in candidates if path in mtimes]
        return sorted(found, key=lambda c: mtimes[c[1]], reverse=True)

    def _read_queue_status(self, adapter: str, state: str, path: Path) -> QueueStatus | None:
        """Build a QueueStatus from a manifest file's metadata.

        Returns None if the file is gone (moved by a runner meanwhile) or, for a finished
        file found by name prefix, belongs to another queue key.
        """
        from ..manifest.loader import load_manifest_metadata
        from ..schema import QueueStatus

        try:
            manifest = load_manifest_metadata(path)
        except FileNotFoundError:
            return None
        except (yaml.YAMLError, ValueError, OSError) as e:
            self._lg.warning("unreadable queue manifest", extra={"path": str(path), "exception": e})
            # Pending and running use exact names; completed and failed use prefix matching,
            # so without their metadata they cannot be attributed to this key.
            return (
                QueueStatus(adapter=adapter, state=state)
                if state in ("pending", "running")
                else None
            )

        if queue_key(manifest.adapter, manifest.series) != adapter:
            return None
        return QueueStatus(
            adapter=manifest.adapter,
            state=state,
            created_at=manifest.created_at,
            output=manifest.output,
        )

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
        key = queue_key(manifest.adapter, manifest.series)
        completed_file = self.completed_path / f"{key}-{suffix}.yaml.gz"
        if manifest.commit:
            # A series build's sets stay in the lineage store, not in every completed manifest
            stored = Manifest({**manifest, "data": Data(format="inline")})
            save_manifest(stored, completed_file, compress=True)
            self._lineage.record_build(manifest.adapter, manifest.commit, "completed", md5=md5)
        else:
            save_manifest(manifest, completed_file, compress=True)

        if md5:
            self._link_manifest_to_adapter(manifest.adapter, md5, completed_file)

        self._remove_queue_source(manifest)
        self._lg.info("completed manifest", extra={"adapter": manifest.adapter, "md5": md5})

    def _remove_queue_source(self, manifest: Manifest) -> None:
        """Remove the queue file the manifest was loaded from, if it is in pending/ or running/.

        Only that file: a newer submission for the same key, pending while this one ran,
        stays queued.
        """
        source = manifest.get("source_path")
        if source is None:
            return
        source = Path(source).resolve()
        queue_dirs = (self.pending_path.resolve(), self.running_path.resolve())
        if source.parent in queue_dirs:
            source.unlink(missing_ok=True)

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
        for lineage_key in ("series", "commit"):
            if result.get(lineage_key):
                config[lineage_key] = result.get(lineage_key)
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
            # Lock: undeploy + rmtree atomically, so a concurrent deploy cannot create a
            # symlink to the version we are about to delete.
            with self._deploy_lock(key):
                self._remove_version_unlocked(key, version_id)
            self._lg.info("removed adapter version", extra={"key": key, "version": version_id})
        else:
            shutil.rmtree(key_path)
            self._lg.info("removed adapter", extra={"key": key})

    def _remove_version_unlocked(self, key: str, version_id: str) -> None:
        """Remove a specific version. Caller must hold _deploy_lock(key)."""
        version_path = self.adapters_path / key / version_id
        config = self._read_adapter_config(key, version_id)
        version_md5 = config.get("md5")
        if version_md5 and self.is_deployed(key, version_md5):
            self.undeploy_adapter(key, version_md5)
        shutil.rmtree(version_path)

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

        # Lock: link + policy removal atomically, so a concurrent drop/remove cannot delete
        # the version we are about to deploy.
        with self._deploy_lock(key):
            # Re-check: version could have been deleted between resolution and lock acquisition
            version_path = self.adapters_path / key / version_id
            if not version_path.exists():
                raise ValueError(f"Version '{version_id}' was removed for adapter '{key}'")
            symlink_path = self._atomic_deploy_symlink(key, version_id, md5)
            if policy == "replace":
                for symlink in self._get_deployed_symlinks(key):
                    if symlink != symlink_path:
                        symlink.unlink()
                        self._lg.info("removed deployment", extra={"symlink": symlink.name})
        self._lg.info(
            "deployed adapter",
            extra={"key": key, "version": version_id, "md5": md5, "policy": policy},
        )

    def _atomic_deploy_symlink(self, key: str, version_id: str, md5: str) -> Path:
        """Create a versioned deploy symlink atomically. Caller must hold _deploy_lock."""
        symlink_path = self.deployed_path / f"{key}-{md5}"
        tmp_link = self.deployed_path / f".{key}-{md5}.{uuid.uuid4().hex}.tmp"
        tmp_link.symlink_to(Path("..") / "adapters" / key / version_id)
        try:
            os.replace(tmp_link, symlink_path)
        finally:
            tmp_link.unlink(missing_ok=True)
        return symlink_path

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
        self.validate_queue_key(adapter)
        work_dir = self.work_path / adapter

        if clean and work_dir.exists():
            self._lg.info("cleaning work area", extra={"adapter": adapter})
            shutil.rmtree(work_dir)

        work_dir.mkdir(parents=True, exist_ok=True)
        return work_dir

    def cleanup_work_area(self, adapter: str) -> None:
        """Clean up a training work area."""
        self.validate_queue_key(adapter)
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

    def validate_queue_key(self, key: str) -> None:
        """Validate a queue key: an adapter key, or ``<adapter>+<series>`` (see queue_key)."""
        for part in key.split(SERIES_SEPARATOR, 1):
            self.validate_key(part)

    # =========================================================================
    # Filesystem Helpers (not in ABC - for backwards compatibility)
    # =========================================================================

    def ensure_directories(self) -> None:
        """Create all required registry directories."""
        for path in (
            self.pending_path,
            self.running_path,
            self.completed_path,
            self.failed_path,
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
        """Get latest version ID for an adapter.

        Versions built by a series other than "main" or the active one are skipped: such
        a series is deployed only by commit (deploy_series), so "latest" (and deploying
        without a version) means the newest version of a series the key serves.
        """
        key_path = self.adapters_path / key
        if not key_path.is_dir():
            return None

        active = self.active_series(key)
        served = {None, DEFAULT_SERIES, active.series if active else None}
        versions = sorted((p.name for p in key_path.iterdir() if p.is_dir()), reverse=True)
        for version_id in versions:
            # Always admit the currently deployed version, regardless of which series built it
            if active and version_id == active.version_id:
                return version_id
            series = self._read_adapter_config(key, version_id).get("series")
            if series in served:
                return version_id
        return None

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
