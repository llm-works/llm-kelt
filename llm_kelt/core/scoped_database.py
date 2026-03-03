"""Database wrapper scoped to a specific PostgreSQL schema."""

from collections.abc import Generator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from appinfra.db.pg.scoped import ScopedPG
    from appinfra.log import Logger
    from sqlalchemy.orm import Session


class ScopedDatabase:
    """
    Database interface scoped to a specific PostgreSQL schema.

    Wraps appinfra's ScopedPG to provide session-level schema isolation.
    Sessions automatically have search_path set to the target schema.

    Usage:
        # From Database.scoped()
        scoped_db = db.scoped("my_schema")

        with scoped_db.session() as session:
            session.add(Record(...))  # Goes to my_schema.* tables
    """

    def __init__(self, lg: "Logger", scoped_pg: "ScopedPG") -> None:
        self._lg = lg
        self._scoped_pg = scoped_pg

    @contextmanager
    def session(self) -> Generator["Session", None, None]:
        """
        Context manager for database sessions scoped to this schema.

        Sessions have search_path set to this schema at creation.
        Automatically commits on success, rolls back on exception.

        Yields:
            SQLAlchemy session configured for this schema
        """
        with self._scoped_pg.session() as session:
            yield session

    def ensure_schema(self) -> None:
        """
        Create the PostgreSQL schema if it doesn't exist.

        Idempotent - safe to call multiple times.

        Raises:
            DatabaseError: If parent PG is in readonly mode
        """
        self._scoped_pg.ensure_schema()

    @property
    def schema(self) -> str:
        """Get the schema name for this scoped database."""
        return self._scoped_pg.schema

    @property
    def engine(self) -> Any:
        """Get the underlying SQLAlchemy engine."""
        return self._scoped_pg.engine
