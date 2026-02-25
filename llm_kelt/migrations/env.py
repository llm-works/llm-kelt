"""Alembic environment configuration for Learn framework."""

import os
import sys
from pathlib import Path

from alembic import context
from sqlalchemy import inspect, text
from sqlalchemy.exc import OperationalError

# Add project root to path (migrations is now inside llm_kelt package)
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from appinfra.config import Config  # noqa: E402
from appinfra.db.pg import PG  # noqa: E402
from appinfra.log import LogConfig, LoggerFactory  # noqa: E402

from llm_kelt.core.models import Base  # noqa: E402

# Load configuration
config_path = os.environ.get("LEARN_CONFIG", str(project_root / "etc" / "infra.yaml"))
db_key = os.environ.get("LEARN_DB_KEY", "main")

app_config = Config(config_path)
db_config = app_config.dbs[db_key]

# This is the Alembic Config object
config = context.config

# Set URL from our config
config.set_main_option("sqlalchemy.url", db_config.url)

# Create logger for migrations
_log_config = LogConfig.from_params(level="info")
_lg = LoggerFactory.create_root(_log_config)


def _get_pg() -> PG:
    """Create appinfra PG instance."""
    return PG(_lg, db_config)


# SQLAlchemy metadata for autogenerate
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode (generates SQL without connecting)."""
    _lg.info("running offline migration (SQL generation mode)")
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()
    _lg.info("offline migration complete")


def _bootstrap_fresh_database(pg: PG) -> None:
    """Bootstrap a fresh database with schema from models."""
    from alembic.script import ScriptDirectory

    _lg.info("bootstrapping fresh database")

    # Use appinfra to create db, extensions, and tables
    pg.migrate(Base)

    # Stamp with head revision
    script = ScriptDirectory.from_config(config)
    head_rev = script.get_current_head()
    _lg.info("stamping database with head revision", extra={"revision": head_rev})

    with pg.engine.connect() as conn:
        conn.execute(
            text("CREATE TABLE IF NOT EXISTS alembic_version (version_num VARCHAR(32) PRIMARY KEY)")
        )
        conn.execute(
            text("INSERT INTO alembic_version (version_num) VALUES (:rev) ON CONFLICT DO NOTHING"),
            {"rev": head_rev},
        )
        conn.commit()

    _lg.info("bootstrap complete")


def run_migrations_online() -> None:  # cq: exempt
    """Run migrations in 'online' mode."""
    from alembic.script import ScriptDirectory

    _lg.info("starting online migration", extra={"db_key": db_key})
    pg = _get_pg()

    # Check if database needs bootstrapping (pg.migrate handles db creation)
    try:
        with pg.engine.connect() as connection:
            inspector = inspect(connection)
            tables = inspector.get_table_names()

            if "alembic_version" not in tables:
                _lg.info("no alembic_version table found, bootstrapping database")
                _bootstrap_fresh_database(pg)
                return

            # Get current and target revisions
            result = connection.execute(text("SELECT version_num FROM alembic_version"))
            current_rev = result.scalar()

            script = ScriptDirectory.from_config(config)
            head_rev = script.get_current_head()

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

            context.configure(connection=connection, target_metadata=target_metadata)
            with context.begin_transaction():
                context.run_migrations()
                connection.commit()
                _lg.info("migration transaction committed")

            # Verify the migration succeeded
            result = connection.execute(text("SELECT version_num FROM alembic_version"))
            new_rev = result.scalar()
            _lg.info("migration complete", extra={"new_revision": new_rev})

    except OperationalError as e:
        # Database doesn't exist - bootstrap it (PostgreSQL error code 3D000)
        if e.orig is not None and hasattr(e.orig, "pgcode") and e.orig.pgcode == "3D000":
            _lg.info("database does not exist, bootstrapping")
            _bootstrap_fresh_database(pg)
        else:
            _lg.error("migration failed", extra={"exception": e})
            raise


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
