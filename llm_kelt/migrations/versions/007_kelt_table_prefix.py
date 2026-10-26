# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-kelt Authors

"""Prefix every kelt table with kelt_.

Tables, indexes and sequences share one namespace per Postgres schema, so kelt's
names collide with those of any other library or application in the same schema.
Renames each kelt table to kelt_<name>, including the embedding tables kelt creates
at runtime (embeddings_[{prefix}_]{dims}_{format}), and the objects named after it:

- idx_/ix_/uq_ indexes and constraints: kelt_ after the prefix (idx_content_created
  -> idx_kelt_content_created)
- primary keys, foreign keys, NOT NULL constraints and sequences: the names Postgres
  generates for the renamed table (kelt_content_pkey, kelt_content_id_seq)

The names are read from the catalog of the current schema. A name no rule covers,
or a new name over Postgres's 63-character limit (Postgres would truncate it), fails
the revision. The version table keeps its name.

Revision ID: 007
Revises: 006
Create Date: 2026-10-25
"""

import re
from collections.abc import Callable, Sequence

from alembic import op
from sqlalchemy import Connection, text

revision: str = "007"
down_revision: str = "006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_PREFIX = "kelt_"

_TABLES = frozenset(
    {
        "atomic_directive_details",
        "atomic_fact_relationships",
        "atomic_facts",
        "atomic_feedback_details",
        "atomic_interaction_details",
        "atomic_prediction_details",
        "atomic_preference_details",
        "atomic_solution_details",
        "content",
        "conv_sessions",
        "kg_entities",
        "kg_entity_aliases",
        "kg_entity_refs",
        "kg_entity_relationships",
        "kg_fact_entities",
    }
)

_EMBEDDING_TABLE = re.compile(r"^embeddings_(?:\w+_)?\d+_(?:f32|f16|i8|i4)$")
_EMBEDDING_COLUMNS = frozenset({"entity_type", "entity_id", "model_name"})

_NAME_PREFIXES = ("idx_", "ix_", "uq_")
_MAX_IDENTIFIER = 63

_TABLES_SQL = text(
    "SELECT c.relname, array_agg(a.attname::text) FROM pg_class c "
    "JOIN pg_attribute a ON a.attrelid = c.oid AND a.attnum > 0 AND NOT a.attisdropped "
    "WHERE c.relnamespace = to_regnamespace(current_schema()) AND c.relkind IN ('r', 'p') "
    "GROUP BY c.relname"
)

# Constraints with the columns they cover, in key order
_CONSTRAINTS_SQL = text(
    "SELECT k.conname, k.contype::text, "
    "array(SELECT a.attname::text FROM unnest(k.conkey) WITH ORDINALITY AS u(attnum, n) "
    "JOIN pg_attribute a ON a.attrelid = k.conrelid AND a.attnum = u.attnum ORDER BY u.n) "
    "FROM pg_constraint k WHERE k.conrelid = CAST(:table AS regclass)"
)

# Indexes not backing a constraint; those are renamed with their constraint
_INDEXES_SQL = text(
    "SELECT i.relname FROM pg_index x JOIN pg_class i ON i.oid = x.indexrelid "
    "WHERE x.indrelid = CAST(:table AS regclass) AND NOT EXISTS ("
    "SELECT 1 FROM pg_constraint k WHERE k.conindid = x.indexrelid "
    "AND k.conrelid = x.indrelid AND k.contype IN ('p', 'u', 'x'))"
)

# Sequences owned by a column (serial or identity)
_SEQUENCES_SQL = text(
    "SELECT s.relname, a.attname FROM pg_depend d "
    "JOIN pg_class s ON s.oid = d.objid AND s.relkind = 'S' "
    "JOIN pg_attribute a ON a.attrelid = d.refobjid AND a.attnum = d.refobjsubid "
    "WHERE d.refobjid = CAST(:table AS regclass) AND d.deptype IN ('a', 'i')"
)


def _quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _checked(name: str) -> str:
    if len(name) > _MAX_IDENTIFIER:
        raise RuntimeError(
            f"{name!r} exceeds Postgres's {_MAX_IDENTIFIER}-character identifier limit"
        )
    return name


def _is_kelt_table(name: str, columns: set[str]) -> bool:
    if name in _TABLES:
        return True
    return bool(_EMBEDDING_TABLE.match(name)) and _EMBEDDING_COLUMNS <= columns


def _add_prefix(name: str) -> str:
    """idx_content_created -> idx_kelt_content_created."""
    for prefix in _NAME_PREFIXES:
        if name.startswith(prefix):
            return _checked(f"{prefix}{_PREFIX}{name.removeprefix(prefix)}")
    raise RuntimeError(f"no rename rule for {name!r}")


def _remove_prefix(name: str) -> str:
    """idx_kelt_content_created -> idx_content_created."""
    for prefix in _NAME_PREFIXES:
        if name.startswith(prefix + _PREFIX):
            return prefix + name.removeprefix(prefix + _PREFIX)
    raise RuntimeError(f"no rename rule for {name!r}")


def _constraint_name(
    table: str, name: str, kind: str, columns: list[str], explicit: Callable[[str], str]
) -> str:
    """Postgres's generated name for primary/foreign keys and NOT NULL, else explicit()."""
    if kind == "p":
        return _checked(f"{table}_pkey")
    if kind == "f":
        return _checked(f"{table}_{'_'.join(columns)}_fkey")
    if kind == "n":
        return _checked(f"{table}_{columns[0]}_not_null")
    return explicit(name)


def _rename_to(statement: str, new: str) -> None:
    """Run ``statement`` (ending in RENAME or RENAME CONSTRAINT <name>) with TO <new>."""
    op.execute(f"{statement} TO {_quote(new)}")


def _rename_objects(
    conn: Connection, schema: str, table: str, explicit: Callable[[str], str]
) -> None:
    """Rename the constraints, indexes and sequences of ``table`` (already renamed)."""
    qualified = f"{_quote(schema)}.{_quote(table)}"
    params = {"table": qualified}
    for name, kind, columns in conn.execute(_CONSTRAINTS_SQL, params).all():
        new = _constraint_name(table, name, kind, columns, explicit)
        if new != name:
            _rename_to(f"ALTER TABLE {qualified} RENAME CONSTRAINT {_quote(name)}", new)
    for (name,) in conn.execute(_INDEXES_SQL, params).all():
        _rename_to(f"ALTER INDEX {_quote(schema)}.{_quote(name)} RENAME", explicit(name))
    for name, column in conn.execute(_SEQUENCES_SQL, params).all():
        new = _checked(f"{table}_{column}_seq")
        if new != name:
            _rename_to(f"ALTER SEQUENCE {_quote(schema)}.{_quote(name)} RENAME", new)


def _rename(
    table_name: Callable[[str, set[str]], str | None], explicit: Callable[[str], str]
) -> None:
    """Rename each table ``table_name`` maps to a new name, then the objects named after it."""
    conn = op.get_bind()
    schema = conn.execute(text("SELECT current_schema()")).scalar_one()
    for old, columns in conn.execute(_TABLES_SQL).all():
        new = table_name(old, set(columns))
        if new is None:
            continue
        _rename_to(f"ALTER TABLE {_quote(schema)}.{_quote(old)} RENAME", _checked(new))
        _rename_objects(conn, schema, new, explicit)


def _upgrade_table(name: str, columns: set[str]) -> str | None:
    return _PREFIX + name if _is_kelt_table(name, columns) else None


def _downgrade_table(name: str, columns: set[str]) -> str | None:
    old = name.removeprefix(_PREFIX)
    return old if name.startswith(_PREFIX) and _is_kelt_table(old, columns) else None


def upgrade() -> None:
    _rename(_upgrade_table, _add_prefix)


def downgrade() -> None:
    _rename(_downgrade_table, _remove_prefix)
