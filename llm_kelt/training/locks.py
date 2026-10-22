# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-kelt Authors

"""One write lock per registry.

Every operation that writes a registry (its queue, the content-addressed store, build
records, submit dates, annotations, the deployed-series record, adapter versions and
deployed links) holds the registry's lock for its whole run, so writers take turns; a
write takes seconds at most, as training runs outside the lock. Readers do not lock: every
write replaces a file atomically.

The lock is an ``flock`` on one file, so it excludes other processes and other threads. It
is reentrant within a thread, so an operation holding it may call another one. Lock order:
the registry's lock before any lock of the content-addressed store.
"""

from __future__ import annotations

import fcntl
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

# Lock files this thread holds, with their hold depth. Module-wide rather than per
# RegistryLock, so two objects on one registry in one thread do not deadlock each other.
_held = threading.local()


class RegistryLock:
    """Exclusive, thread-reentrant write lock of a registry, as one lock file."""

    def __init__(self, path: Path) -> None:
        """Initialize the lock.

        Args:
            path: Lock file, outside every directory the lock guards, so deleting guarded
                state never deletes the lock file.
        """
        self._path = path

    @contextmanager
    def hold(self) -> Iterator[None]:
        """Hold the lock for the block; a nested hold in the same thread passes through."""
        path = self._path.resolve()
        depth: dict[Path, int] = _held.__dict__.setdefault("depth", {})
        if depth.get(path):
            depth[path] += 1
            try:
                yield
            finally:
                depth[path] -= 1
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            depth[path] = 1
            try:
                yield
            finally:
                del depth[path]
