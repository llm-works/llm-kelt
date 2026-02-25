"""Core infrastructure for Kelt framework.

Provides database, models, exceptions, and shared utilities.
Memory models (facts, predictions, etc.) are in memory/atomic.
"""

# Base utilities
from .base import Base, utc_now

# Models
from .content import Content, ContentStore

# Database
from .database import Database
from .embedding import Embedding, EmbeddingStore

# Exceptions
from .exceptions import (
    ConfigurationError,
    DatabaseError,
    KeltError,
    NotFoundError,
    SchemaVersionError,
    ValidationError,
)

# Schema management
from .schema import SchemaManager, SchemaState, SchemaStatus

# Types
from .types import PagedResult, ScoredEntity

__all__ = [
    # Base
    "Base",
    "utc_now",
    # Database
    "Database",
    # Models
    "Content",
    "ContentStore",
    "Embedding",
    "EmbeddingStore",
    # Types
    "ScoredEntity",
    "PagedResult",
    # Schema
    "SchemaManager",
    "SchemaState",
    "SchemaStatus",
    # Exceptions
    "KeltError",
    "ValidationError",
    "NotFoundError",
    "DatabaseError",
    "ConfigurationError",
    "SchemaVersionError",
]
