# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-kelt Authors

"""Custom exceptions for Kelt framework."""


class KeltError(Exception):
    """Base exception for Kelt framework."""

    pass


class ValidationError(KeltError):
    """Raised when input validation fails."""

    pass


class NotFoundError(KeltError):
    """Raised when a requested resource is not found."""

    pass


class DatabaseError(KeltError):
    """Raised when a database operation fails."""

    pass


class ConfigError(KeltError):
    """Raised when configuration is invalid or missing."""

    pass


class SchemaVersionError(KeltError):
    """Raised when schema version is incompatible (e.g., newer than library)."""

    pass


class ConflictError(KeltError):
    """Raised when an operation conflicts with existing state."""

    pass


class ContextOverflowError(KeltError):
    """Raised when conversation exceeds max_tokens and compaction cannot reduce it.

    This typically happens when:
    - A single message (e.g., large tool result) exceeds max_tokens
    - The preserved messages (system + min_recent_messages) exceed max_tokens

    Attributes:
        token_count: Current token count after compaction attempt.
        max_tokens: Configured maximum tokens.
    """

    def __init__(self, message: str, token_count: int, max_tokens: int) -> None:
        super().__init__(message)
        self.token_count = token_count
        self.max_tokens = max_tokens
