"""Core utility functions for Learn framework."""

from datetime import UTC, datetime


def utc_now() -> datetime:
    """Return current UTC time as timezone-aware datetime."""
    return datetime.now(UTC)
