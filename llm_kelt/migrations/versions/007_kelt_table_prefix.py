# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-kelt Authors

"""Prefix every kelt table with kelt_, and align the 006 layout with the models.

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

Before any change, each kelt table present is checked against the columns revision
006 gave it (_LAYOUT_006): a missing column or a different type means the table is
not kelt's, and fails the revision. Extra columns are allowed.

The layout changes:

- kelt_content.id becomes bigint, the type of the content_id columns referencing it
  (rewrites the table under an exclusive lock)
- indexes whose columns lead another index of the same table are dropped
  (_REDUNDANT_INDEXES, and idx_<table>_entity of each embedding table)
- kelt_conv_sessions is created where missing: schemas set up while its model was not
  imported have none. The downgrade keeps it, renamed to conv_sessions.

Revision ID: 007
Revises: 006
Create Date: 2026-10-25
"""

import re
from collections.abc import Callable, Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy import Connection, text
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "007"
down_revision: str = "006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_PREFIX = "kelt_"

# Columns of each kelt table at revision 006: table, column, type (format_type)
_LAYOUT_006 = """
atomic_directive_details id bigint
atomic_directive_details fact_id bigint
atomic_directive_details directive_type character varying(50)
atomic_directive_details parsed_rules jsonb
atomic_directive_details status character varying(20)
atomic_directive_details expires_at timestamp with time zone
atomic_fact_relationships id bigint
atomic_fact_relationships source_id bigint
atomic_fact_relationships target_id bigint
atomic_fact_relationships relationship_type character varying(50)
atomic_fact_relationships confidence double precision
atomic_fact_relationships extra jsonb
atomic_fact_relationships context_key character varying(255)
atomic_fact_relationships created_at timestamp with time zone
atomic_facts id bigint
atomic_facts context_key character varying(255)
atomic_facts type character varying(20)
atomic_facts content text
atomic_facts content_hash character varying(64)
atomic_facts category character varying(50)
atomic_facts source character varying(50)
atomic_facts confidence double precision
atomic_facts active boolean
atomic_facts created_at timestamp with time zone
atomic_facts updated_at timestamp with time zone
atomic_feedback_details id bigint
atomic_feedback_details fact_id bigint
atomic_feedback_details content_id bigint
atomic_feedback_details signal character varying(20)
atomic_feedback_details strength double precision
atomic_feedback_details feedback_at timestamp with time zone
atomic_feedback_details provider_type character varying(50)
atomic_feedback_details provider character varying(100)
atomic_feedback_details tags character varying(50)[]
atomic_feedback_details comment text
atomic_feedback_details context jsonb
atomic_interaction_details id bigint
atomic_interaction_details fact_id bigint
atomic_interaction_details content_id bigint
atomic_interaction_details interaction_type character varying(50)
atomic_interaction_details duration_ms integer
atomic_interaction_details scroll_depth double precision
atomic_interaction_details context jsonb
atomic_prediction_details id bigint
atomic_prediction_details fact_id bigint
atomic_prediction_details resolution_type character varying(50)
atomic_prediction_details resolution_date date
atomic_prediction_details resolution_event text
atomic_prediction_details resolution_metric jsonb
atomic_prediction_details verification_source character varying(100)
atomic_prediction_details verification_url text
atomic_prediction_details status character varying(20)
atomic_prediction_details outcome character varying(20)
atomic_prediction_details outcome_confidence double precision
atomic_prediction_details actual_result text
atomic_prediction_details tags character varying(50)[]
atomic_prediction_details resolved_at timestamp with time zone
atomic_preference_details id bigint
atomic_preference_details fact_id bigint
atomic_preference_details context text
atomic_preference_details chosen text
atomic_preference_details rejected text
atomic_preference_details margin double precision
atomic_preference_details extra jsonb
atomic_solution_details id bigint
atomic_solution_details fact_id bigint
atomic_solution_details agent_name character varying(100)
atomic_solution_details problem text
atomic_solution_details problem_context jsonb
atomic_solution_details answer jsonb
atomic_solution_details answer_text text
atomic_solution_details tokens_used integer
atomic_solution_details latency_ms integer
atomic_solution_details tool_calls jsonb
content id integer
content context_key character varying(255)
content external_id character varying(255)
content source character varying(100)
content url text
content title text
content content_text text
content content_hash character varying(64)
content extra jsonb
content created_at timestamp with time zone
content fetched_at timestamp with time zone
conv_sessions id integer
conv_sessions session_id character varying(255)
conv_sessions messages jsonb
conv_sessions token_count integer
conv_sessions config jsonb
conv_sessions extra jsonb
conv_sessions created_at timestamp with time zone
conv_sessions updated_at timestamp with time zone
kg_entities id bigint
kg_entities scope_key character varying(255)
kg_entities canonical_name character varying(255)
kg_entities entity_type character varying(50)
kg_entities description text
kg_entities extra jsonb
kg_entities created_at timestamp with time zone
kg_entities updated_at timestamp with time zone
kg_entity_aliases id bigint
kg_entity_aliases entity_id bigint
kg_entity_aliases scope_key character varying(255)
kg_entity_aliases alias character varying(255)
kg_entity_aliases alias_normalized character varying(255)
kg_entity_aliases extra jsonb
kg_entity_aliases created_at timestamp with time zone
kg_entity_refs id bigint
kg_entity_refs entity_id bigint
kg_entity_refs scope_key character varying(255)
kg_entity_refs source_type character varying(50)
kg_entity_refs source_id character varying(255)
kg_entity_refs source_url character varying(2048)
kg_entity_refs snippet text
kg_entity_refs sentiment double precision
kg_entity_refs extra jsonb
kg_entity_refs ref_at timestamp with time zone
kg_entity_relationships id bigint
kg_entity_relationships scope_key character varying(255)
kg_entity_relationships from_entity_id bigint
kg_entity_relationships to_entity_id bigint
kg_entity_relationships relationship_type character varying(50)
kg_entity_relationships confidence double precision
kg_entity_relationships extra jsonb
kg_entity_relationships created_at timestamp with time zone
kg_fact_entities fact_id bigint
kg_fact_entities entity_id bigint
kg_fact_entities scope_key character varying(255)
kg_fact_entities role character varying(50)
kg_fact_entities confidence double precision
kg_fact_entities extra jsonb
kg_fact_entities created_at timestamp with time zone
"""


def _parse_layout(layout: str) -> dict[str, dict[str, str]]:
    tables: dict[str, dict[str, str]] = {}
    for line in layout.strip().splitlines():
        table, column, type_ = line.split(" ", 2)
        tables.setdefault(table, {})[column] = type_
    return tables


_TABLES = _parse_layout(_LAYOUT_006)

# Embedding tables vary in layout by format and over time; matched by name and the
# columns every format has
_EMBEDDING_TABLE = re.compile(r"^embeddings_(?:\w+_)?\d+_(?:f32|f16|i8|i4)$")
_EMBEDDING_COLUMNS = frozenset({"entity_type", "entity_id", "model_name"})

# Indexes whose columns lead another index of the same table (a primary key's, a unique
# constraint's or a composite one's): index -> (table, columns) at revision 006
_REDUNDANT_INDEXES = {
    "idx_atomic_directive_fact": ("atomic_directive_details", ("fact_id",)),
    "idx_atomic_facts_context": ("atomic_facts", ("context_key",)),
    "ix_atomic_facts_context_key": ("atomic_facts", ("context_key",)),
    "idx_atomic_feedback_fact": ("atomic_feedback_details", ("fact_id",)),
    "idx_atomic_interaction_fact": ("atomic_interaction_details", ("fact_id",)),
    "idx_atomic_prediction_fact": ("atomic_prediction_details", ("fact_id",)),
    "idx_atomic_preference_fact": ("atomic_preference_details", ("fact_id",)),
    "idx_atomic_solution_fact": ("atomic_solution_details", ("fact_id",)),
    "idx_content_context": ("content", ("context_key",)),
    "ix_content_context_key": ("content", ("context_key",)),
    "idx_conv_sessions_session_id": ("conv_sessions", ("session_id",)),
    "ix_kg_entities_scope": ("kg_entities", ("scope_key",)),
    "ix_kg_entity_aliases_scope": ("kg_entity_aliases", ("scope_key",)),
    "ix_kg_entity_refs_entity": ("kg_entity_refs", ("entity_id",)),
    "ix_kg_entity_relationships_from": ("kg_entity_relationships", ("from_entity_id",)),
    "ix_kg_entity_relationships_scope": ("kg_entity_relationships", ("scope_key",)),
    "ix_kg_fact_entities_fact": ("kg_fact_entities", ("fact_id",)),
}

# Each embedding table's idx_<table>_entity leads its unique (entity_type, entity_id,
# model_name) index
_EMBEDDING_ENTITY_INDEX = ("entity_type", "entity_id")

_NAME_PREFIXES = ("idx_", "ix_", "uq_")
_MAX_IDENTIFIER = 63

_COLUMNS_SQL = text(
    "SELECT c.relname, a.attname::text, format_type(a.atttypid, a.atttypmod) FROM pg_class c "
    "JOIN pg_attribute a ON a.attrelid = c.oid AND a.attnum > 0 AND NOT a.attisdropped "
    "WHERE c.relnamespace = to_regnamespace(current_schema()) AND c.relkind IN ('r', 'p')"
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


def _columns(conn: Connection) -> dict[str, dict[str, str]]:
    """Column name -> type of each table in the current schema."""
    tables: dict[str, dict[str, str]] = {}
    for table, column, type_ in conn.execute(_COLUMNS_SQL).all():
        tables.setdefault(table, {})[column] = type_
    return tables


def _check_layout(conn: Connection) -> None:
    """Fail unless each kelt table present has the columns revision 006 gave it."""
    actual = _columns(conn)
    for table, expected in _TABLES.items():
        if table not in actual:
            continue
        wrong = sorted(f"{c} {t}" for c, t in expected.items() if actual[table].get(c) != t)
        if wrong:
            raise RuntimeError(f"{table!r} is not kelt's table; it lacks {', '.join(wrong)}")


def _is_kelt_table(name: str, columns: set[str]) -> bool:
    if name in _TABLES:
        return True
    return bool(_EMBEDDING_TABLE.match(name)) and columns >= _EMBEDDING_COLUMNS


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
    conn: Connection,
    schema: str,
    table_name: Callable[[str, set[str]], str | None],
    explicit: Callable[[str], str],
) -> None:
    """Rename each table ``table_name`` maps to a new name, then the objects named after it."""
    for old, columns in _columns(conn).items():
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


def _redundant_indexes(conn: Connection) -> list[tuple[str, str, tuple[str, ...]]]:
    """(index, table, columns) of each redundant index, named as at revision 006."""
    indexes: list[tuple[str, str, tuple[str, ...]]] = [
        (index, table, columns) for index, (table, columns) in _REDUNDANT_INDEXES.items()
    ]
    for table, columns in _columns(conn).items():
        if _EMBEDDING_TABLE.match(table) and set(columns) >= _EMBEDDING_COLUMNS:
            indexes.append((f"idx_{table}_entity", table, _EMBEDDING_ENTITY_INDEX))
    return indexes


def _set_content_id_type(conn: Connection, schema: str, type_: str) -> None:
    """Change kelt_content.id and its sequence to ``type_``."""
    table = f"{_quote(schema)}.{_quote(_PREFIX + 'content')}"
    op.execute(f"ALTER TABLE {table} ALTER COLUMN id TYPE {type_}")
    sequence = conn.execute(
        text("SELECT pg_get_serial_sequence(:table, 'id')"), {"table": table}
    ).scalar_one()
    # A sequence at its type's default bounds moves to the bounds of the new type
    op.execute(f"ALTER SEQUENCE {sequence} AS {type_}")


def _create_conv_sessions(conn: Connection, schema: str) -> None:
    """Create kelt_conv_sessions as the models define it, unless it exists."""
    table = _PREFIX + "conv_sessions"
    exists = conn.execute(
        text("SELECT to_regclass(:table)"), {"table": f"{_quote(schema)}.{_quote(table)}"}
    ).scalar()
    if exists is not None:
        return
    op.create_table(
        table,
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("session_id", sa.String(255), nullable=False),
        sa.Column("messages", JSONB(), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=False),
        sa.Column("config", JSONB(), nullable=False),
        sa.Column("extra", JSONB(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.UniqueConstraint("session_id", name="uq_kelt_conv_sessions_session_id"),
        schema=schema,
    )
    op.create_index("idx_kelt_conv_sessions_updated_at", table, ["updated_at"], schema=schema)


def upgrade() -> None:
    conn = op.get_bind()
    schema = conn.execute(text("SELECT current_schema()")).scalar_one()
    _check_layout(conn)
    for index, _, _ in _redundant_indexes(conn):
        op.execute(f"DROP INDEX IF EXISTS {_quote(schema)}.{_quote(index)}")
    _rename(conn, schema, _upgrade_table, _add_prefix)
    _set_content_id_type(conn, schema, "bigint")
    _create_conv_sessions(conn, schema)


def downgrade() -> None:
    conn = op.get_bind()
    schema = conn.execute(text("SELECT current_schema()")).scalar_one()
    _set_content_id_type(conn, schema, "integer")
    _rename(conn, schema, _downgrade_table, _remove_prefix)
    for index, table, columns in _redundant_indexes(conn):
        op.execute(
            f"CREATE INDEX {_quote(index)} ON {_quote(schema)}.{_quote(table)} "
            f"({', '.join(map(_quote, columns))})"
        )
