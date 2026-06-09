"""Session storage backends for conversation persistence.

Provides abstract SessionStorage interface and concrete implementations
for file-based and database-backed storage.
"""

from .base import SessionStorage, SessionSummary, StoredSession
from .file import FileSessionStorage

__all__ = [
    "SessionStorage",
    "StoredSession",
    "SessionSummary",
    "FileSessionStorage",
]
