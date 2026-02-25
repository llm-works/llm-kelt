"""Custom exceptions for Learn framework."""


class KeltError(Exception):
    """Base exception for Learn framework."""

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


class ConfigurationError(KeltError):
    """Raised when configuration is invalid or missing."""

    pass


class SchemaVersionError(KeltError):
    """Raised when schema version is incompatible (e.g., newer than library)."""

    pass


class ConflictError(KeltError):
    """Raised when an operation conflicts with existing state."""

    pass
