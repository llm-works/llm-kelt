"""Database wrapper using appinfra's PG class."""

from collections.abc import Generator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import sqlalchemy_utils
from appinfra.db.pg import PG
from appinfra.log import Logger

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from .scoped_database import ScopedDatabase


class Database:
    """
    Database interface for Kelt framework.

    Wraps appinfra's PG class with Kelt-specific configuration
    and model management.

    Usage:
        from appinfra.db.pg import PG
        from appinfra.log import LogConfig, LoggerFactory

        lg = LoggerFactory.create_root(LogConfig.from_params(level="info"))
        pg = PG(lg, db_config)
        db = Database(lg, pg)

        with db.session() as session:
            session.add(Feedback(...))
            session.commit()
    """

    def __init__(self, lg: Logger, pg: PG) -> None:
        self._lg = lg
        self._pg = pg

    @contextmanager
    def session(self) -> Generator["Session", None, None]:
        """
        Context manager for database sessions.

        Automatically commits on success, rolls back on exception.

        Usage:
            with db.session() as session:
                session.add(record)
                # Commits automatically on exit
        """
        session = self._pg.session()
        try:
            yield session
            session.commit()
        except Exception as e:
            self._lg.warning("session rollback", extra={"exception": e})
            session.rollback()
            raise
        finally:
            session.close()

    def ensure_database(self) -> None:
        """Create the database if PG is configured with create_db and it doesn't exist.

        Mirrors the create_db logic from PG.migrate() so callers that bypass
        PG.migrate() (e.g., SchemaManager) still get automatic database creation.
        """
        create_db = getattr(self._pg.cfg, "create_db", False)
        if create_db is True and not sqlalchemy_utils.database_exists(self._pg.engine.url):
            sqlalchemy_utils.create_database(self._pg.engine.url)
            self._lg.info("created database")

    def ensure_pg_schema(self) -> None:
        """Create the PostgreSQL schema if configured and doesn't exist.

        When PG is initialized with schema="some_schema", this ensures the schema
        exists before tables are created. No-op if no schema is configured.
        """
        self._pg.create_schema()

    def configure_schema(self, schema_name: str) -> None:
        """Configure PG with a schema after construction.

        This is used when ClientContext.schema_name is set but PG wasn't
        originally configured with a schema. It dynamically sets up schema
        isolation so all subsequent queries use the specified schema.

        Args:
            schema_name: PostgreSQL schema name to configure.

        Raises:
            ValueError: If PG is already configured with a different schema.
        """
        if self._pg.schema:
            if self._pg.schema != schema_name:
                raise ValueError(
                    f"Cannot reconfigure schema: PG already configured with '{self._pg.schema}'"
                )
            return  # Already configured with this schema

        # Import SchemaManager and configure PG's schema isolation
        from appinfra.db.pg.schema import SchemaManager

        schema_mgr = SchemaManager(self._pg.engine, schema_name, self._lg)
        schema_mgr.setup_listeners()
        self._pg._schema_mgr = schema_mgr  # type: ignore[attr-defined]
        self._lg.debug("configured schema isolation", extra={"schema": schema_name})

    @property
    def engine(self) -> Any:
        """Get SQLAlchemy engine."""
        return self._pg.engine

    @property
    def schema(self) -> str | None:
        """Get the configured PostgreSQL schema name, if any."""
        return self._pg.schema

    def health_check(self) -> dict[str, Any]:
        """Check database connectivity."""
        result: dict[str, Any] = self._pg.health_check()
        return result

    def get_pool_status(self) -> dict[str, Any]:
        """Get connection pool status."""
        result: dict[str, Any] = self._pg.get_pool_status()
        return result

    def scoped(self, schema_name: str) -> "ScopedDatabase":
        """
        Get a database view scoped to a specific schema.

        Sessions from the scoped database have search_path set to the schema.
        This allows a single Database instance to serve multiple schemas.

        Args:
            schema_name: PostgreSQL schema name

        Returns:
            ScopedDatabase bound to the schema

        Example:
            >>> scoped_db = db.scoped("my_schema")
            >>> with scoped_db.session() as session:
            ...     session.query(MyModel).all()  # Uses my_schema.* tables
        """
        from .scoped_database import ScopedDatabase

        return ScopedDatabase(self._lg, self._pg.scoped(schema_name))
