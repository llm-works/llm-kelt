"""Manifest-related errors."""

from __future__ import annotations

from pathlib import Path


class ManifestError(ValueError):
    """Base class for manifest errors."""

    pass


class CorruptedManifestError(ManifestError):
    """Raised when a manifest file exists but is empty or has invalid format."""

    def __init__(self, path: Path, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"Corrupted manifest {path}: {reason}")
