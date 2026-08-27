# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-kelt Authors

"""High-level setup helpers for embedding kelt into a foreign database.

Most consumers want a single call: "given my PG instance, make sure kelt's
schema is ready." `ensure_schema()` provides that entry point without
forcing callers to construct a full `Client` (and invent a `ClientContext`)
just to run migrations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .core.database import Database
from .core.schema import SchemaStatus

if TYPE_CHECKING:
    from appinfra.db.pg import PG
    from appinfra.log import Logger


def ensure_schema(
    lg: Logger,
    pg: PG,
    schema_name: str | None = None,
) -> SchemaStatus:
    """Ensure kelt's schema is ready in the given PG instance.

    Idempotent. Performs the full setup sequence in one call:
        1. Creates the database itself if PG was configured with `create_db=True`
        2. Creates the postgres schema namespace if PG was configured with one
        3. Runs migrations (or creates tables on a fresh database) and stamps version

    Use this when embedding kelt into a foreign database — e.g., a service that
    owns its own PG and wants kelt's tables alongside its own. The migration-setup
    sequence is shared with `Client(schema_mode=SchemaMode.ENSURE)` via `Database.ensure_schema()`,
    so both entry points run the same steps. Note that constructing a `Client`
    afterward still requires passing `schema_name` via `ClientContext` if it
    differs from `pg.schema`; this helper only handles the migration side.

    Args:
        lg: Logger instance.
        pg: appinfra PG instance.
        schema_name: PostgreSQL schema name. If not provided, falls back to
            `pg.schema`, and ultimately to "public" if neither is set.
            Pass this explicitly when embedding into a multi-tenant host to
            avoid silently landing tables in `public`.

    Returns:
        SchemaStatus describing the resulting state. `SchemaStatus` and
        `SchemaState` are re-exported from `llm_kelt` for convenience.

    Example:
        from appinfra.db.pg import PG
        from appinfra.log import LogConfig, LoggerFactory

        import llm_kelt

        lg = LoggerFactory.create_root(LogConfig.from_params(level="info"))
        pg = PG(lg, db_config)
        status = llm_kelt.ensure_schema(lg, pg, schema_name="my_tenant")
    """
    db = Database(lg, pg)
    return db.ensure_schema(schema_name)
