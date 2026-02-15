"""Custom exceptions for Learn framework."""


class LearnError(Exception):
    """Base exception for Learn framework."""

    pass


class ValidationError(LearnError):
    """Raised when input validation fails."""

    pass


class NotFoundError(LearnError):
    """Raised when a requested resource is not found."""

    pass


class DatabaseError(LearnError):
    """Raised when a database operation fails."""

    pass


class ConfigurationError(LearnError):
    """Raised when configuration is invalid or missing."""

    pass


class SchemaVersionError(LearnError):
    """Raised when schema version is incompatible (e.g., newer than library)."""

    pass


class ConflictError(LearnError):
    """Raised when an operation conflicts with existing state."""

    pass
