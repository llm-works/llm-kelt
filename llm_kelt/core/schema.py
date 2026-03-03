"""Schema management for programmatic database initialization.

Provides SchemaManager for safe, concurrent schema initialization
using PostgreSQL advisory locks.
"""

import re
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

# Schema name validation pattern - must be a valid SQL identifier.
# Must start with lowercase letter, can contain lowercase letters, numbers, and underscores.
# Limited to 63 chars (PostgreSQL identifier limit) to prevent truncation.
_SCHEMA_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,62}$")


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
        from llm_kelt.core.schema import SchemaManager

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
        schema_name: str | None = None,
    ) -> None:
        """Initialize SchemaManager.

        Args:
            lg: Logger instance
            engine: SQLAlchemy engine
            migrations_path: Path to migrations directory. Defaults to package migrations dir.
            schema_name: PostgreSQL schema name (for logging). Defaults to "public".

        Raises:
            ValueError: If schema_name is not a valid SQL identifier.
        """
        self._lg = lg
        self._engine = engine
        self._schema_name = schema_name or "public"

        if not _SCHEMA_NAME_PATTERN.match(self._schema_name):
            raise ValueError(
                f"Invalid schema name '{self._schema_name}'. "
                "Must be 1-63 chars, start with a lowercase letter, and contain only "
                "lowercase letters, numbers, and underscores."
            )

        if migrations_path is None:
            # Default: llm_kelt/migrations (inside the package)
            package_root = Path(__file__).parent.parent
            migrations_path = package_root / "migrations"
        self._migrations_path = migrations_path

    def _get_alembic_config(self) -> AlembicConfig:
        """Create Alembic config pointing to our migrations."""
        alembic_ini = self._migrations_path / "alembic.ini"
        config = AlembicConfig(str(alembic_ini))
        config.set_main_option("script_location", str(self._migrations_path))
        config.set_main_option("sqlalchemy.url", str(self._engine.url))
        # Tell Alembic which schema to use for alembic_version table
        config.set_main_option("version_table_schema", self._schema_name)
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
            context = MigrationContext.configure(
                conn, opts={"version_table_schema": self._schema_name}
            )
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
                self._lg.warning("failed to acquire schema lock", extra={"exception": e})
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

    def _set_search_path(self, conn) -> None:
        """Set search_path to target schema for DDL operations.

        Uses SET LOCAL to scope the change to the current transaction,
        preventing search_path leakage on pooled connections.
        """
        conn.execute(text(f'SET LOCAL search_path TO "{self._schema_name}", public'))

    def _create_tables_in_schema(self, conn) -> None:
        """Create all tables in the target schema.

        SQLAlchemy's create_all() checks table existence using the default schema,
        so we must explicitly set each table's schema before creating.
        """
        original_schemas: dict[str, str | None] = {}
        try:
            # Temporarily set schema on all tables
            for table in Base.metadata.tables.values():
                original_schemas[table.name] = table.schema
                table.schema = self._schema_name
            # Create tables in target schema
            Base.metadata.create_all(conn)
        finally:
            # Restore original schemas to avoid side effects
            for table in Base.metadata.tables.values():
                table.schema = original_schemas.get(table.name)

    def _bootstrap_fresh_database(self) -> None:
        """Bootstrap a fresh database with schema from models."""
        self._lg.debug("creating db schema...", extra={"schema": self._schema_name})
        head_version = self._get_head_version()

        with self._engine.connect() as conn:
            self._set_search_path(conn)
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            self._create_tables_in_schema(conn)
            self._stamp_alembic_version(conn, head_version)
            conn.commit()

        self._lg.info(
            "db schema created", extra={"schema": self._schema_name, "version": head_version}
        )

    def _stamp_alembic_version(self, conn, version: str) -> None:
        """Stamp alembic_version table with given revision."""
        conn.execute(
            text("CREATE TABLE IF NOT EXISTS alembic_version (version_num VARCHAR(32) PRIMARY KEY)")
        )
        conn.execute(
            text(
                "INSERT INTO alembic_version (version_num) VALUES (:rev) "
                "ON CONFLICT (version_num) DO NOTHING"
            ),
            {"rev": version},
        )

    def _run_upgrade(self) -> None:
        """Run Alembic upgrade to head."""
        self._lg.info("running schema upgrade to head")
        config = self._get_alembic_config()
        command.upgrade(config, "head")
        self._lg.info("schema upgrade complete")

    def _check_version_compatible(self, status: SchemaStatus) -> None:
        """Raise SchemaVersionError if schema is too new."""
        if status.state == SchemaState.TOO_NEW:
            raise SchemaVersionError(
                f"Database schema version '{status.current_version}' is newer than "
                f"library version '{status.head_version}'. Cannot downgrade. "
                "Please upgrade the llm-kelt package."
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
            extra = {"schema": self._schema_name, "version": status.current_version}
            self._lg.trace("schema already current", extra=extra)
            return status

        self._check_version_compatible(status)
        return self._migrate_with_lock(wait, timeout_seconds)

    def _migrate_with_lock(self, wait: bool, timeout_seconds: float) -> SchemaStatus:
        """Acquire lock and run migration. Connection must stay open for lock duration."""
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
