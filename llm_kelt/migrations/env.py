# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-kelt Authors

"""Alembic environment configuration for Kelt framework.

This env.py supports two modes:
1. Called via SchemaManager (normal app usage): runs on the open connection passed in
   config.attributes["connection"], inside SchemaManager's transaction
2. Called via alembic CLI (development): connects with the URL from the alembic config,
   or else from the KELT_CONFIG env var
"""

import os

from alembic import context
from appinfra.log import LogConfig, LoggerFactory
from sqlalchemy import Connection, create_engine, text
from sqlalchemy.exc import OperationalError, ProgrammingError

from llm_kelt.core.models import Base
from llm_kelt.core.schema import _VERSION_TABLE_NAME

# This is the Alembic Config object
config = context.config

# SchemaManager runs alembic in the caller's process and passes its logger; only the
# standalone alembic CLI, a process of its own, creates one.
_lg = config.attributes.get("lg") or LoggerFactory.create_root(LogConfig.from_params(level="info"))


def _get_database_url() -> str:
    """Get the database URL for the standalone alembic CLI.

    Priority:
    1. URL set in the alembic config (sqlalchemy.url)
    2. Load from KELT_CONFIG file
    """
    url: str | None = config.get_main_option("sqlalchemy.url")
    if url:
        return url

    # Fall back to loading from config file (standalone CLI mode)
    config_path = os.environ.get("KELT_CONFIG")
    if not config_path:
        raise RuntimeError(
            "Database URL not configured. Either:\n"
            "  1. Use SchemaManager.ensure_schema() (recommended)\n"
            "  2. Set KELT_CONFIG env var to config file path for alembic CLI"
        )

    from appinfra.config import Config

    db_key = os.environ.get("KELT_DB_KEY", "main")
    app_config = Config(config_path)
    db_config = app_config.dbs[db_key]
    return str(db_config.url)


# SQLAlchemy metadata for autogenerate
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode (generates SQL without connecting)."""
    _lg.info("running offline migration (SQL generation mode)")
    url = _get_database_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        version_table=_VERSION_TABLE_NAME,
    )

    with context.begin_transaction():
        context.run_migrations()
    _lg.info("offline migration complete")


def _current_revision(connection: Connection, schema_name: str) -> str | None:
    """Read the version row; None while the version table does not exist yet."""
    # Use savepoint to check version without poisoning transaction on failure
    connection.execute(text("SAVEPOINT check_version"))
    try:
        result = connection.execute(
            text(f'SELECT version_num FROM "{schema_name}"."{_VERSION_TABLE_NAME}"')
        )
        current_rev: str | None = result.scalar()
        connection.execute(text("RELEASE SAVEPOINT check_version"))
        return current_rev
    except (OperationalError, ProgrammingError):
        # Table doesn't exist on first run - alembic will create it
        connection.execute(text("ROLLBACK TO SAVEPOINT check_version"))
        return None


def _run_migrations(connection: Connection, schema_name: str) -> None:
    """Run the migrations on ``connection``, inside its owner's transaction."""
    from alembic.script import ScriptDirectory

    # Set search_path for DDL operations in target schema
    connection.execute(text(f'SET LOCAL search_path TO "{schema_name}", public'))

    # Ensure pgvector extension exists (required for embeddings table)
    connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

    head_rev = ScriptDirectory.from_config(config).get_current_head()
    current_rev = _current_revision(connection, schema_name)
    _lg.info("migration state", extra={"current_revision": current_rev, "head_revision": head_rev})
    if current_rev == head_rev:
        _lg.info("database already at head revision, nothing to do")
        return

    _lg.info("running migrations", extra={"from_revision": current_rev, "to_revision": head_rev})
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        version_table_schema=schema_name,
        version_table=_VERSION_TABLE_NAME,
    )
    # The connection is already in a transaction, so alembic starts and commits none
    with context.begin_transaction():
        context.run_migrations()
    new_rev = _current_revision(connection, schema_name)
    _lg.info("migration complete", extra={"new_revision": new_rev})


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    When called via SchemaManager, just runs the migrations on its connection; it
    commits them. SchemaManager handles bootstrapping and version checking separately.
    """
    schema_name = config.get_main_option("version_table_schema") or "public"
    _lg.info("starting online migration", extra={"schema": schema_name})
    try:
        connection = config.attributes.get("connection")
        if connection is not None:
            _run_migrations(connection, schema_name)
            return
        engine = create_engine(_get_database_url())
        try:
            with engine.begin() as conn:
                _run_migrations(conn, schema_name)
        finally:
            engine.dispose()
    except OperationalError as e:
        _lg.error("migration failed", extra={"exception": e})
        raise


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
