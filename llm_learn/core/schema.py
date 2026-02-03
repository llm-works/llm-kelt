"""Schema management for programmatic database initialization.

Provides SchemaManager for safe, concurrent schema initialization
using PostgreSQL advisory locks.
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

from alembic import command
from alembic.config import Config as AlembicConfig
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import text

from .exceptions import SchemaVersionError
from .models import Base

if TYPE_CHECKING:
    from appinfra.log import Logger
    from sqlalchemy.engine import Engine


# Advisory lock key for schema migrations.
# IMPORTANT: This must be a fixed constant, not computed with hash(), because Python's
# hash() uses a randomized seed (since 3.3) that differs between processes.
# Value chosen arbitrarily but must remain stable across all library versions.
_ADVISORY_LOCK_KEY = 7829104563218907456


class SchemaState(Enum):
    """Schema version state relative to library version."""

    MISSING = "missing"  # No alembic_version table
    CURRENT = "current"  # Version matches head
    NEEDS_UPGRADE = "upgrade"  # Schema behind, can upgrade
    TOO_NEW = "too_new"  # Schema ahead, cannot downgrade


@dataclass
class SchemaStatus:
    """Current schema status information."""

    state: SchemaState
    current_version: str | None
    head_version: str


class SchemaManager:
    """Manages database schema initialization and migrations.

    Provides thread-safe schema management using PostgreSQL advisory locks
    for concurrent agent support.

    Usage:
        from appinfra.log import LogConfig, LoggerFactory
        from llm_learn.core.schema import SchemaManager

        lg = LoggerFactory.create_root(LogConfig.from_params(level="info"))
        manager = SchemaManager(lg, engine)

        # Check current state
        status = manager.get_status()

        # Initialize/migrate (blocks until lock acquired)
        status = manager.ensure_schema()
    """

    def __init__(
        self,
        lg: "Logger",
        engine: "Engine",
        migrations_path: Path | None = None,
    ) -> None:
        """Initialize SchemaManager.

        Args:
            lg: Logger instance
            engine: SQLAlchemy engine
            migrations_path: Path to migrations directory. Defaults to package migrations dir.
        """
        self._lg = lg
        self._engine = engine

        if migrations_path is None:
            # Default: llm_learn/migrations (inside the package)
            package_root = Path(__file__).parent.parent
            migrations_path = package_root / "migrations"
        self._migrations_path = migrations_path

    def _get_alembic_config(self) -> AlembicConfig:
        """Create Alembic config pointing to our migrations."""
        alembic_ini = self._migrations_path / "alembic.ini"
        config = AlembicConfig(str(alembic_ini))
        config.set_main_option("script_location", str(self._migrations_path))
        config.set_main_option("sqlalchemy.url", str(self._engine.url))
        return config

    def _get_head_version(self) -> str:
        """Get the head (latest) migration revision."""
        config = self._get_alembic_config()
        script = ScriptDirectory.from_config(config)
        head = script.get_current_head()
        if head is None:
            raise SchemaVersionError("No migrations found in migrations directory")
        return str(head)

    def _get_current_version(self) -> str | None:
        """Get current database schema version, or None if not initialized."""
        with self._engine.connect() as conn:
            context = MigrationContext.configure(conn)
            revision = context.get_current_revision()
            return str(revision) if revision is not None else None

    def _is_version_in_chain(self, version: str) -> bool:
        """Check if a version exists in our migration chain."""
        config = self._get_alembic_config()
        script = ScriptDirectory.from_config(config)
        try:
            # walk_revisions yields all revisions from base to head
            for rev in script.walk_revisions():
                if rev.revision == version:
                    return True
            return False
        except Exception:
            return False

    def get_status(self) -> SchemaStatus:
        """Get current schema status.

        Returns:
            SchemaStatus with state, current version, and head version
        """
        head_version = self._get_head_version()
        current_version = self._get_current_version()

        if current_version is None:
            state = SchemaState.MISSING
        elif current_version == head_version:
            state = SchemaState.CURRENT
        elif self._is_version_in_chain(current_version):
            state = SchemaState.NEEDS_UPGRADE
        else:
            # Version not in our chain - likely newer than library
            state = SchemaState.TOO_NEW

        return SchemaStatus(
            state=state,
            current_version=current_version,
            head_version=head_version,
        )

    def _acquire_lock(self, conn, wait: bool, timeout_seconds: float) -> bool:
        """Acquire advisory lock for schema operations.

        IMPORTANT: The lock is session-level, meaning it's held until the connection
        closes or pg_advisory_unlock is called. The caller must keep `conn` open
        for the duration of the protected operation.

        Args:
            conn: SQLAlchemy connection to acquire lock on (must stay open!)
            wait: If True, block until lock acquired. If False, return immediately.
            timeout_seconds: Max time to wait if blocking.

        Returns:
            True if lock acquired, False if not (only when wait=False or timeout)
        """
        if wait:
            # Set statement timeout and use blocking lock
            conn.execute(text(f"SET LOCAL statement_timeout = '{int(timeout_seconds * 1000)}ms'"))
            try:
                conn.execute(text(f"SELECT pg_advisory_lock({_ADVISORY_LOCK_KEY})"))
                return True
            except Exception as e:
                self._lg.warning("Failed to acquire schema lock", extra={"exception": e})
                return False
        else:
            # Non-blocking attempt
            result = conn.execute(
                text(f"SELECT pg_try_advisory_lock({_ADVISORY_LOCK_KEY})")
            ).scalar()
            return bool(result)

    def _release_lock(self, conn) -> None:
        """Release advisory lock.

        Args:
            conn: The same connection that acquired the lock.
        """
        conn.execute(text(f"SELECT pg_advisory_unlock({_ADVISORY_LOCK_KEY})"))
        conn.commit()

    def _bootstrap_fresh_database(self) -> None:
        """Bootstrap a fresh database with schema from models."""
        self._lg.info("Bootstrapping fresh database schema")

        # Create pgvector extension before creating tables (required for Vector columns)
        with self._engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.commit()

        # Create all tables from SQLAlchemy models
        Base.metadata.create_all(self._engine)

        # Stamp with head revision
        head_version = self._get_head_version()
        with self._engine.connect() as conn:
            conn.execute(
                text(
                    "CREATE TABLE IF NOT EXISTS alembic_version "
                    "(version_num VARCHAR(32) PRIMARY KEY)"
                )
            )
            conn.execute(
                text(
                    "INSERT INTO alembic_version (version_num) VALUES (:rev) "
                    "ON CONFLICT (version_num) DO NOTHING"
                ),
                {"rev": head_version},
            )
            conn.commit()

        self._lg.info("Database schema bootstrapped", extra={"version": head_version})

    def _run_upgrade(self) -> None:
        """Run Alembic upgrade to head."""
        self._lg.info("Running schema upgrade to head")
        config = self._get_alembic_config()
        command.upgrade(config, "head")
        self._lg.info("Schema upgrade complete")

    def _check_version_compatible(self, status: SchemaStatus) -> None:
        """Raise SchemaVersionError if schema is too new."""
        if status.state == SchemaState.TOO_NEW:
            raise SchemaVersionError(
                f"Database schema version '{status.current_version}' is newer than "
                f"library version '{status.head_version}'. Cannot downgrade. "
                "Please upgrade the llm-learn package."
            )

    def _apply_migration(self, status: SchemaStatus) -> None:
        """Apply migration based on current state."""
        if status.state == SchemaState.MISSING:
            self._bootstrap_fresh_database()
        elif status.state == SchemaState.NEEDS_UPGRADE:
            self._run_upgrade()

    def ensure_schema(
        self,
        wait: bool = True,
        timeout_seconds: float = 30.0,
    ) -> SchemaStatus:
        """Ensure database schema is current.

        Thread-safe schema initialization using PostgreSQL advisory locks.
        Safe to call from multiple concurrent processes/threads.

        Args:
            wait: If True, block until lock acquired. If False, fail fast.
            timeout_seconds: Max time to wait for lock (only used if wait=True).

        Returns:
            SchemaStatus after operation

        Raises:
            SchemaVersionError: If schema is newer than library (TOO_NEW state)
            TimeoutError: If lock cannot be acquired within timeout
        """
        status = self.get_status()

        # Fast path: already current
        if status.state == SchemaState.CURRENT:
            self._lg.debug("Schema already current", extra={"version": status.current_version})
            return status

        self._check_version_compatible(status)

        # Acquire lock for modification.
        # IMPORTANT: We must keep this connection open for the duration of the migration,
        # because pg_advisory_lock is session-level and releases when the connection closes.
        with self._engine.connect() as conn:
            if not self._acquire_lock(conn, wait, timeout_seconds):
                raise TimeoutError(
                    f"Could not acquire schema lock within {timeout_seconds}s. "
                    "Another process may be running migrations."
                )

            try:
                # Re-check under lock (another process may have migrated)
                status = self.get_status()
                if status.state == SchemaState.CURRENT:
                    return status

                self._check_version_compatible(status)
                self._apply_migration(status)
                return self.get_status()
            finally:
                self._release_lock(conn)
