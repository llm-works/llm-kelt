"""Database wrapper using appinfra's PG class."""

from collections.abc import Generator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from appinfra.config import Config
from appinfra.db.pg import PG
from appinfra.log import LogConfig, LoggerFactory

from .models import Base

if TYPE_CHECKING:
    from sqlalchemy.orm import Session


class Database:
    """
    Database interface for Learn framework.

    Wraps appinfra's PG class with Learn-specific configuration
    and model management.

    Usage:
        db = Database.from_config("etc/infra.yaml")
        with db.session() as session:
            session.add(Feedback(...))
            session.commit()
    """

    def __init__(self, pg: PG) -> None:
        self._pg = pg

    @classmethod
    def from_config(
        cls,
        config_path: str,
        db_key: str = "main",
        log_level: str = "info",
    ) -> "Database":
        """
        Create Database from configuration file.

        Args:
            config_path: Path to YAML config file
            db_key: Database configuration key (default: "main")
            log_level: Logging level

        Returns:
            Configured Database instance
        """
        config = Config(config_path)
        db_config = config.dbs[db_key]

        if db_config is None:
            raise ValueError(f"Database config 'dbs.{db_key}' not found in {config_path}")

        # Create logger
        log_config = LogConfig.from_params(level=log_level)
        logger = LoggerFactory.create_root(log_config)

        # Create PG instance
        pg = PG(logger, db_config)

        return cls(pg)

    @classmethod
    def from_pg(cls, pg: PG) -> "Database":
        """Create Database from existing PG instance."""
        return cls(pg)

    def migrate(self) -> None:
        """Run migrations to create all tables from SQLAlchemy models."""
        # Extensions (e.g., vector) are auto-created via pg.yaml config
        self._pg.migrate(Base)

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
        except Exception:
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
