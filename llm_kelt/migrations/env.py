"""Alembic environment configuration for Kelt framework.

This env.py supports two modes:
1. Called via SchemaManager (normal app usage): URL is pre-set in alembic config
2. Called via alembic CLI (development): loads config from KELT_CONFIG env var
"""

import os

from alembic import context
from appinfra.log import LogConfig, LoggerFactory
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError, ProgrammingError

from llm_kelt.core.models import Base

# This is the Alembic Config object
config = context.config

# Create logger for migrations
_log_config = LogConfig.from_params(level="info")
_lg = LoggerFactory.create_root(_log_config)


def _get_database_url() -> str:
    """Get database URL from alembic config or fall back to app config file.

    Priority:
    1. URL already set in alembic config (by SchemaManager)
    2. Load from KELT_CONFIG file (for standalone alembic CLI)
    """
    url = config.get_main_option("sqlalchemy.url")
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
    )

    with context.begin_transaction():
        context.run_migrations()
    _lg.info("offline migration complete")


def run_migrations_online() -> None:  # cq: exempt
    """Run migrations in 'online' mode.

    When called via SchemaManager, just runs the migrations.
    SchemaManager handles bootstrapping and version checking separately.
    """
    from alembic.script import ScriptDirectory

    url = _get_database_url()
    schema_name = config.get_main_option("version_table_schema") or "public"
    _lg.info("starting online migration", extra={"schema": schema_name})

    engine = create_engine(url)

    try:
        with engine.connect() as connection:
            # Set search_path for DDL operations in target schema
            connection.execute(text(f'SET LOCAL search_path TO "{schema_name}", public'))

            # Ensure pgvector extension exists (required for embeddings table)
            connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

            # Get current and target revisions for logging
            script = ScriptDirectory.from_config(config)
            head_rev = script.get_current_head()

            # Use savepoint to check version without poisoning transaction on failure
            connection.execute(text("SAVEPOINT check_version"))
            try:
                result = connection.execute(
                    text(f'SELECT version_num FROM "{schema_name}".alembic_version')
                )
                current_rev = result.scalar()
                connection.execute(text("RELEASE SAVEPOINT check_version"))
            except (OperationalError, ProgrammingError):
                # Table doesn't exist on first run - alembic will create it
                connection.execute(text("ROLLBACK TO SAVEPOINT check_version"))
                current_rev = None

            _lg.info(
                "migration state",
                extra={"current_revision": current_rev, "head_revision": head_rev},
            )

            if current_rev == head_rev:
                _lg.info("database already at head revision, nothing to do")
                return

            _lg.info(
                "running migrations",
                extra={"from_revision": current_rev, "to_revision": head_rev},
            )

            context.configure(
                connection=connection,
                target_metadata=target_metadata,
                version_table_schema=schema_name,
            )
            with context.begin_transaction():
                context.run_migrations()
                connection.commit()
                _lg.info("migration transaction committed")

            # Verify the migration succeeded
            result = connection.execute(
                text(f'SELECT version_num FROM "{schema_name}".alembic_version')
            )
            new_rev = result.scalar()
            _lg.info("migration complete", extra={"new_revision": new_rev})

    except OperationalError as e:
        _lg.error("migration failed", extra={"exception": e})
        raise
    finally:
        engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
