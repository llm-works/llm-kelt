# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-kelt Authors

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
