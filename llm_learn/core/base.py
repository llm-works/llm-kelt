"""Base classes and utilities for all models."""

import hashlib
from datetime import UTC, datetime

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Base class for all Learn models."""

    pass


def utc_now() -> datetime:
    """Return current UTC time as timezone-aware datetime."""
    return datetime.now(UTC)


def generate_id(prefix: str, *parts: str | None) -> str:
    """
    Generate a hash-based ID from path components.

    Creates a globally unique, deterministic ID by hashing the prefix and parts.
    The ID is computed once at entity creation and remains stable even if
    the entity's slug or other attributes change.

    Args:
        prefix: Entity type prefix (e.g., "domain", "workspace", "profile").
                Prevents ID collisions across entity types.
        parts: Path components to include in hash. None values become empty strings.

    Returns:
        32-character hex string (128 bits / 16 bytes).

    Examples:
        >>> generate_id("domain", "acme")
        'a3f8b2c1d4e5f6781b2c3d4e5f6a7b8c'

        >>> generate_id("workspace", "acme", "research")
        '1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e'

        >>> generate_id("profile", None, "default", "me")  # domain is None
        'f6781b2c3d4e5f6a7b8c9d0e1f2a3b4c'
    """
    # Build path string: prefix:part1:part2:...
    path = f"{prefix}:{':'.join(p or '' for p in parts)}"
    # SHA-256 hash, truncated to 16 bytes (32 hex chars)
    return hashlib.sha256(path.encode()).digest()[:16].hex()
