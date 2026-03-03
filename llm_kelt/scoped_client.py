"""Client operations scoped to a specific PostgreSQL schema."""

from __future__ import annotations

from typing import TYPE_CHECKING

from appinfra.log import Logger

from .core.embedding import EmbeddingStore
from .core.schema import SchemaManager
from .memory import atomic

if TYPE_CHECKING:
    from .client import Client
    from .core.scoped_database import ScopedDatabase
    from .memory.atomic import Protocol


class ScopedClient:
    """
    Client operations scoped to a specific PostgreSQL schema.

    Provides lazy initialization: the schema and tables are created
    on first use if ensure_schema was True at construction.

    Usage:
        # Get scoped client from parent
        scoped = client.with_schema("my_schema")

        # First operation triggers lazy initialization
        scoped.atomic.solutions.record(...)
    """

    def __init__(
        self,
        lg: Logger,
        parent: Client,
        schema_name: str,
        ensure_schema: bool,
    ) -> None:
        """
        Initialize scoped client.

        Args:
            lg: Logger instance
            parent: Parent Client for shared resources (embedder, etc.)
            schema_name: PostgreSQL schema name for this scope
            ensure_schema: If True, create schema + tables on first use
        """
        self._lg = lg
        self._parent = parent
        self._schema_name = schema_name
        self._ensure_schema = ensure_schema

        # Lazy-initialized
        self._scoped_db: ScopedDatabase | None = None
        self._atomic: Protocol | None = None
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Lazy initialization: create schema + tables on first use."""
        if self._initialized:
            return

        self._scoped_db = self._parent._db.scoped(self._schema_name)

        if self._ensure_schema:
            # Create PostgreSQL schema if needed
            self._scoped_db.ensure_schema()

            # Run Alembic migrations for this schema
            manager = SchemaManager(self._lg, self._scoped_db.engine, schema_name=self._schema_name)
            manager.ensure_schema()

        # Create stores with scoped database
        embedding_store = EmbeddingStore(self._scoped_db.session)
        self._atomic = atomic.Protocol(
            self._lg,
            self._scoped_db.session,
            self._parent._context.context_key,
            embedder=self._parent._embedder,
            embedding_store=embedding_store,
        )
        self._initialized = True

    @property
    def atomic(self) -> Protocol:
        """Access atomic memory protocol scoped to this schema."""
        self._ensure_initialized()
        assert self._atomic is not None  # Guaranteed by _ensure_initialized
        return self._atomic

    @property
    def schema_name(self) -> str:
        """Get the schema name for this scoped client."""
        return self._schema_name
