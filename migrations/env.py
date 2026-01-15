"""Alembic environment configuration for Learn framework."""

import os
import sys
from pathlib import Path

from alembic import context
from sqlalchemy import inspect, text
from sqlalchemy.exc import OperationalError

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from appinfra.config import Config  # noqa: E402
from appinfra.db.pg import PG  # noqa: E402
from appinfra.log import LogConfig, LoggerFactory  # noqa: E402

from llm_learn.core.models import Base  # noqa: E402

# Load configuration
config_path = os.environ.get("LEARN_CONFIG", str(project_root / "etc" / "infra.yaml"))
db_key = os.environ.get("LEARN_DB_KEY", "main")

app_config = Config(config_path)
db_config = app_config.dbs[db_key]

# This is the Alembic Config object
config = context.config

# Set URL from our config
config.set_main_option("sqlalchemy.url", db_config.url)


def _get_pg() -> PG:
    """Create appinfra PG instance (deferred to avoid early connection)."""
    log_config = LogConfig.from_params(level="warning")
    logger = LoggerFactory.create_root(log_config)
    return PG(logger, db_config)


# SQLAlchemy metadata for autogenerate
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def _bootstrap_fresh_database(pg: PG) -> None:
    """Bootstrap a fresh database with schema from models."""
    from alembic.script import ScriptDirectory

    # Use appinfra to create db, extensions, and tables
    pg.migrate(Base)

    # Stamp with head revision
    script = ScriptDirectory.from_config(config)
    head_rev = script.get_current_head()
    with pg.engine.connect() as conn:
        conn.execute(
            text("CREATE TABLE IF NOT EXISTS alembic_version (version_num VARCHAR(32) PRIMARY KEY)")
        )
        conn.execute(
            text("INSERT INTO alembic_version (version_num) VALUES (:rev) ON CONFLICT DO NOTHING"),
            {"rev": head_rev},
        )
        conn.commit()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    pg = _get_pg()

    # Check if database needs bootstrapping (pg.migrate handles db creation)
    try:
        with pg.engine.connect() as connection:
            inspector = inspect(connection)
            if "alembic_version" not in inspector.get_table_names():
                _bootstrap_fresh_database(pg)
                return

            context.configure(connection=connection, target_metadata=target_metadata)
            with context.begin_transaction():
                context.run_migrations()
    except OperationalError as e:
        # Database doesn't exist - bootstrap it (PostgreSQL error code 3D000)
        if hasattr(e.orig, "pgcode") and e.orig.pgcode == "3D000":
            _bootstrap_fresh_database(pg)
        else:
            raise


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
