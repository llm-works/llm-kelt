"""Database wrapper using appinfra's PG class."""

from collections.abc import Generator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import sqlalchemy_utils
from appinfra.db.pg import PG
from appinfra.log import Logger

if TYPE_CHECKING:
    from sqlalchemy.orm import Session


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

    @property
    def engine(self) -> Any:
        """Get SQLAlchemy engine."""
        return self._pg.engine

    def health_check(self) -> dict[str, Any]:
        """Check database connectivity."""
        result: dict[str, Any] = self._pg.health_check()
        return result

    def get_pool_status(self) -> dict[str, Any]:
        """Get connection pool status."""
        result: dict[str, Any] = self._pg.get_pool_status()
        return result
