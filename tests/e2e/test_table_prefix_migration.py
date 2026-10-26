# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-kelt Authors

"""Tests for revision 007, which prefixes every kelt table with kelt_."""

from collections.abc import Iterator
from contextlib import contextmanager
from functools import partial
from pathlib import Path

import pytest
from alembic import command
from sqlalchemy import text
from sqlalchemy.orm import Session

from llm_kelt.core.schema import _VERSION_TABLE_NAME, SchemaManager
from llm_kelt.embedding.factory import Factory
from llm_kelt.embedding.types import Config, QuantizationFormat

_SCHEMA = "kelt_prefix_migration_test"

# Catalog of a schema at revision 006, captured from the models of that revision
_CATALOG_006 = Path(__file__).with_name("schema_006_catalog.tsv")

_CATALOG_SQL = text("""
SELECT 'table' AS kind, c.relname AS name, '' AS owner, '' AS detail FROM pg_class c
  WHERE c.relnamespace = to_regnamespace(:schema) AND c.relkind = 'r'
UNION ALL
SELECT 'column', a.attname, c.relname, format_type(a.atttypid, a.atttypmod)
    || CASE WHEN a.attnotnull THEN ' NOT NULL' ELSE '' END
    || coalesce(' DEFAULT ' || replace(pg_get_expr(d.adbin, d.adrelid), :schema || '.', ''), '')
  FROM pg_attribute a JOIN pg_class c ON c.oid = a.attrelid
  LEFT JOIN pg_attrdef d ON d.adrelid = a.attrelid AND d.adnum = a.attnum
  WHERE c.relnamespace = to_regnamespace(:schema) AND c.relkind = 'r' AND a.attnum > 0
    AND NOT a.attisdropped
UNION ALL
SELECT 'index', i.relname, t.relname, replace(pg_get_indexdef(i.oid), :schema || '.', '')
  FROM pg_index x JOIN pg_class i ON i.oid = x.indexrelid JOIN pg_class t ON t.oid = x.indrelid
  WHERE t.relnamespace = to_regnamespace(:schema)
UNION ALL
SELECT 'con_' || k.contype::text, k.conname, t.relname,
    replace(pg_get_constraintdef(k.oid), :schema || '.', '')
  FROM pg_constraint k JOIN pg_class t ON t.oid = k.conrelid
  WHERE t.relnamespace = to_regnamespace(:schema)
UNION ALL
SELECT 'seq', s.relname, coalesce(t.relname, ''), format_type(q.seqtypid, NULL)
  FROM pg_class s JOIN pg_sequence q ON q.seqrelid = s.oid
  LEFT JOIN pg_depend d ON d.objid = s.oid AND d.classid = 'pg_class'::regclass
    AND d.deptype IN ('a', 'i')
  LEFT JOIN pg_class t ON t.oid = d.refobjid
  WHERE s.relnamespace = to_regnamespace(:schema) AND s.relkind = 'S'
""")


def _catalog(engine, not_null: bool = False) -> set[tuple[str, ...]]:
    """(kind, name, table, definition) of every table, column, index, constraint, sequence.

    NOT NULL constraints are named only from Postgres 18 on; ``not_null`` includes them.
    """
    with engine.connect() as conn:
        rows = conn.execute(_CATALOG_SQL, {"schema": _SCHEMA}).all()
    return {
        tuple(row)
        for row in rows
        if _VERSION_TABLE_NAME not in (row.name, row.owner) and (not_null or row.kind != "con_n")
    }


def _catalog_006() -> set[tuple[str, ...]]:
    lines = _CATALOG_006.read_text().splitlines()
    return {tuple(line.split("\t")) for line in lines if line and not line.startswith("#")}


def _version(engine) -> str:
    with engine.connect() as conn:
        return conn.execute(
            text(f'SELECT version_num FROM "{_SCHEMA}"."{_VERSION_TABLE_NAME}"')
        ).scalar_one()


def _downgrade_to_006(manager: SchemaManager, engine) -> None:
    config = manager._get_alembic_config()
    with engine.begin() as conn:
        config.attributes["connection"] = conn
        command.downgrade(config, "006")


def _execute(engine, sql: str) -> None:
    with engine.begin() as conn:
        conn.execute(text(f'SET LOCAL search_path TO "{_SCHEMA}"'))
        conn.execute(text(sql))


@contextmanager
def _session(engine) -> Iterator[Session]:
    """A session committed on exit, as the stores expect of their session factory."""
    with Session(engine) as session:
        yield session
        session.commit()


def _fresh_schema(manager: SchemaManager, engine) -> None:
    """A schema bootstrapped at head, with two embedding tables and a row in each."""
    with engine.begin() as conn:
        conn.execute(text(f'DROP SCHEMA IF EXISTS "{_SCHEMA}" CASCADE'))
        conn.execute(text(f'CREATE SCHEMA "{_SCHEMA}"'))
    manager.ensure_schema()
    for fmt, prefix in ((QuantizationFormat.F32, None), (QuantizationFormat.I8, "docs")):
        config = Config(context_key="test", format=fmt, dimensions=8, prefix=prefix, schema=_SCHEMA)
        client = Factory().create(partial(_session, engine), config)
        client.store("atomic.fact", "1", [0.5] * 8, "model-a")
    _execute(
        engine,
        "INSERT INTO kelt_atomic_facts "
        "(type, content, content_hash, source, confidence, active, created_at) "
        "VALUES ('assertion', 'fact one', 'h1', 'user', 1.0, true, now())",
    )


@pytest.fixture
def schema(logger, pg_with_tables):
    """A fresh schema at head and its SchemaManager; dropped afterwards."""
    engine = pg_with_tables.engine
    manager = SchemaManager(logger, engine, schema_name=_SCHEMA)
    _fresh_schema(manager, engine)
    yield manager, engine
    with engine.begin() as conn:
        conn.execute(text(f'DROP SCHEMA IF EXISTS "{_SCHEMA}" CASCADE'))


@pytest.mark.xdist_group("schema")
class TestKeltTablePrefix:
    """Revision 007 against a schema laid out as revision 006 left it."""

    def test_downgrade_restores_006_catalog(self, schema):
        """The downgrade yields exactly the catalog of a schema at 006."""
        manager, engine = schema
        _downgrade_to_006(manager, engine)

        assert _version(engine) == "006"
        assert _catalog(engine) == _catalog_006()

    @pytest.mark.parametrize(
        "setup_sql",
        [
            pytest.param(None, id="all-tables"),
            # Schemas set up while the session model was not imported have no such table
            pytest.param("DROP TABLE conv_sessions", id="without-conv-sessions"),
        ],
    )
    def test_upgrade_matches_fresh_schema(self, schema, setup_sql):
        """Upgrading from 006 yields the catalog of a schema bootstrapped at head, data intact."""
        manager, engine = schema
        fresh = _catalog(engine, not_null=True)
        _downgrade_to_006(manager, engine)
        if setup_sql:
            _execute(engine, setup_sql)

        status = manager.ensure_schema()

        assert status.current_version == "007"
        assert _catalog(engine, not_null=True) == fresh
        with engine.begin() as conn:
            facts = conn.execute(text(f'SELECT content FROM "{_SCHEMA}".kelt_atomic_facts'))
            embeddings = conn.execute(
                text(f'SELECT count(*) FROM "{_SCHEMA}".kelt_embeddings_docs_8_i8')
            )
            assert facts.scalars().all() == ["fact one"]
            assert embeddings.scalar_one() == 1
            new_id = conn.execute(
                text(
                    f'INSERT INTO "{_SCHEMA}".kelt_atomic_facts '
                    "(type, content, content_hash, source, confidence, active, created_at) "
                    "VALUES ('assertion', 'fact two', 'h2', 'user', 1.0, true, now()) "
                    "RETURNING id"
                )
            ).scalar_one()
        assert new_id == 2

    @pytest.mark.parametrize(
        ("setup_sql", "error"),
        [
            pytest.param(
                "CREATE INDEX extra_content_source ON content (source)",
                "no rename rule for 'extra_content_source'",
                id="unknown-name",
            ),
            pytest.param(
                f'CREATE TABLE "embeddings_{"p" * 40}_8_f32" (id bigint PRIMARY KEY, '
                "entity_type varchar, entity_id varchar, model_name varchar)",
                "63-character identifier limit",
                id="name-too-long",
            ),
            pytest.param(
                "DROP TABLE content CASCADE; CREATE TABLE content (id serial PRIMARY KEY, body text)",
                "'content' is not kelt's table",
                id="foreign-table",
            ),
            pytest.param(
                "ALTER TABLE content ALTER COLUMN source TYPE text",
                r"'content' is not kelt's table; it lacks source character varying\(100\)",
                id="column-type",
            ),
        ],
    )
    def test_upgrade_refuses_and_rolls_back(self, schema, setup_sql, error):
        """A table or name 007 cannot account for fails the whole upgrade."""
        manager, engine = schema
        _downgrade_to_006(manager, engine)
        _execute(engine, setup_sql)
        before = _catalog(engine)

        with pytest.raises(RuntimeError, match=error):
            manager.ensure_schema()

        assert _version(engine) == "006"
        assert _catalog(engine) == before
