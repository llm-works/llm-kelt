"""Database wrapper using appinfra's PG class."""

from collections.abc import Generator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from appinfra.db.pg import PG
from appinfra.log import Logger

if TYPE_CHECKING:
    from sqlalchemy.orm import Session


class Database:
    """
    Database interface for Learn framework.

    Wraps appinfra's PG class with Learn-specific configuration
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
