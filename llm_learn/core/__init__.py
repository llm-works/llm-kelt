"""Core infrastructure for Learn framework.

Provides database, models, exceptions, and shared utilities.
Memory models (facts, predictions, etc.) are in memory/atomic.
"""

# Base utilities
from .base import Base, generate_id, utc_now

# Models
from .content import Content, ContentStore

# Database
from .database import Database
from .domain import Domain
from .embedding import Embedding, EmbeddingStore

# Exceptions
from .exceptions import (
    ConfigurationError,
    DatabaseError,
    LearnError,
    NotFoundError,
    SchemaVersionError,
    ValidationError,
)
from .profile import Profile

# Schema management
from .schema import SchemaManager, SchemaState, SchemaStatus

# Types
from .types import PagedResult, ScoredEntity
from .workspace import Workspace

__all__ = [
    # Base
    "Base",
    "generate_id",
    "utc_now",
    # Database
    "Database",
    # Models
    "Domain",
    "Workspace",
    "Profile",
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
    "LearnError",
    "ValidationError",
    "NotFoundError",
    "DatabaseError",
    "ConfigurationError",
    "SchemaVersionError",
]
