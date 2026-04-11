"""High-level setup helpers for embedding kelt into a foreign database.

Most consumers want a single call: "given my PG instance, make sure kelt's
schema is ready." The verbose form requires chaining `Database.ensure_database()`,
`Database.ensure_pg_schema()`, and `SchemaManager.ensure_schema()` — and reaching
into `core.*` modules that are not part of the public API.

`ensure_schema()` collapses that into one call.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .core.database import Database
from .core.schema import SchemaManager, SchemaStatus

if TYPE_CHECKING:
    from appinfra.db.pg import PG
    from appinfra.log import Logger


def ensure_schema(lg: Logger, pg: PG) -> SchemaStatus:
    """Ensure kelt's schema is ready in the given PG instance.

    Idempotent. Performs the full setup sequence in one call:
        1. Creates the database itself if PG was configured with `create_db=True`
        2. Creates the postgres schema namespace if PG was configured with one
        3. Runs migrations (or creates tables on a fresh database) and stamps version

    Use this when embedding kelt into a foreign database — e.g., a service that
    owns its own PG and wants kelt's tables alongside its own.

    Args:
        lg: Logger instance.
        pg: appinfra PG instance. Schema name is taken from `pg.schema`
            (defaults to "public" if unset).

    Returns:
        SchemaStatus describing the resulting state.

    Example:
        from appinfra.db.pg import PG
        from appinfra.log import LogConfig, LoggerFactory

        import llm_kelt

        lg = LoggerFactory.create_root(LogConfig.from_params(level="info"))
        pg = PG(lg, db_config)
        llm_kelt.ensure_schema(lg, pg)
    """
    db = Database(lg, pg)
    db.ensure_database()
    db.ensure_pg_schema()
    mgr = SchemaManager(lg, pg.engine, schema_name=pg.schema or "public")
    return mgr.ensure_schema()
