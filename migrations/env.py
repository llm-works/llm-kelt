"""Alembic environment configuration for Learn framework."""

import os
import sys
from pathlib import Path

from alembic import context
from sqlalchemy.engine import Engine

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from appinfra.config import Config  # noqa: E402
from appinfra.db.pg import create_engine_from_config  # noqa: E402

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

# SQLAlchemy metadata for autogenerate
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL and not an Engine,
    though an Engine is acceptable here as well. By skipping the Engine
    creation we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine and associate a
    connection with the context.
    """
    # Create engine from config
    connectable: Engine = create_engine_from_config(db_config)

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
