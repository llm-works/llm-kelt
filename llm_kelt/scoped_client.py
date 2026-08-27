# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-kelt Authors

"""Client operations scoped to a specific PostgreSQL schema."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

from appinfra.log import Logger

from .core.errors import SchemaVersionError
from .core.schema import SchemaManager, SchemaMode, SchemaState
from .memory import atomic

if TYPE_CHECKING:
    from .client import Client
    from .core.scoped_database import ScopedDatabase
    from .memory.atomic import Protocol


class ScopedClient:
    """
    Client operations scoped to a specific PostgreSQL schema.

    Provides lazy initialization: the schema and tables are created
    on first use when schema_mode is SchemaMode.ENSURE.

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
        schema_mode: SchemaMode,
    ) -> None:
        """
        Initialize scoped client.

        Args:
            lg: Logger instance
            parent: Parent Client for shared resources (embedder, etc.)
            schema_name: PostgreSQL schema name for this scope
            schema_mode: Inherited from parent. ENSURE creates schema + tables
                on first use; VERIFY/SKIP assume the schema exists.
        """
        self._lg = lg
        self._parent = parent
        self._schema_name = schema_name
        self._schema_mode = schema_mode

        # Lazy-initialized
        self._scoped_db: ScopedDatabase | None = None
        self._atomic: Protocol | None = None
        self._initialized = False
        self._init_lock = threading.Lock()

    def _do_initialize(self) -> None:
        """Perform actual initialization (called once, inside lock)."""
        self._scoped_db = self._parent._db.scoped(self._schema_name)

        if self._schema_mode is SchemaMode.ENSURE:
            self._scoped_db.ensure_schema()
            manager = SchemaManager(self._lg, self._scoped_db.engine, schema_name=self._schema_name)
            manager.ensure_schema()
        elif self._schema_mode is SchemaMode.VERIFY:
            manager = SchemaManager(self._lg, self._scoped_db.engine, schema_name=self._schema_name)
            status = manager.get_status()
            if status.state != SchemaState.CURRENT:
                raise SchemaVersionError(
                    f"Scoped schema '{self._schema_name}' is not current "
                    f"(state={status.state.value}). Use schema_mode=SchemaMode.ENSURE."
                )

        self._atomic = atomic.Protocol(
            self._lg,
            self._scoped_db.session,
            self._parent._context.context_key,
            embedder=self._parent._embedder,
            embedding_factory=self._parent._embedding_factory,
            embedding_format=self._parent._embedding_format,
            embedding_dimensions=self._parent._embedding_dimensions,
            embedding_schema=self._schema_name,
        )
        self._initialized = True

    def _ensure_initialized(self) -> None:
        """Lazy initialization: create schema + tables on first use.

        Thread-safe via double-checked locking pattern.
        """
        if self._initialized:
            return

        with self._init_lock:
            if self._initialized:
                return
            self._do_initialize()

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
