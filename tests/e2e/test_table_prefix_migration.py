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

# Registers the conversation session model, so the bootstrap creates its table
from llm_kelt.conversation.storage import db as _conversation_db  # noqa: F401
from llm_kelt.core.schema import _VERSION_TABLE_NAME, SchemaManager
from llm_kelt.embedding.factory import Factory
from llm_kelt.embedding.types import Config, QuantizationFormat

_SCHEMA = "kelt_prefix_migration_test"

# Names of a schema at revision 006, captured from the models of that revision
_NAMES_006 = Path(__file__).with_name("schema_006_names.txt")

_NAMES_SQL = text("""
SELECT 'table' AS kind, c.relname AS name, '' AS owner FROM pg_class c
  WHERE c.relnamespace = to_regnamespace(:schema) AND c.relkind = 'r'
UNION ALL
SELECT 'index', i.relname, t.relname FROM pg_index x
  JOIN pg_class i ON i.oid = x.indexrelid JOIN pg_class t ON t.oid = x.indrelid
  WHERE t.relnamespace = to_regnamespace(:schema)
UNION ALL
SELECT 'con_' || k.contype::text, k.conname, t.relname FROM pg_constraint k
  JOIN pg_class t ON t.oid = k.conrelid WHERE t.relnamespace = to_regnamespace(:schema)
UNION ALL
SELECT 'seq', relname, '' FROM pg_class
  WHERE relnamespace = to_regnamespace(:schema) AND relkind = 'S'
""")


def _names(engine, not_null: bool = False) -> set[tuple[str, ...]]:
    """(kind, name, table) of every table, index, constraint and sequence in the schema.

    NOT NULL constraints are named only from Postgres 18 on; ``not_null`` includes them.
    """
    with engine.connect() as conn:
        rows = conn.execute(_NAMES_SQL, {"schema": _SCHEMA}).all()
    return {
        tuple(filter(None, row))
        for row in rows
        if _VERSION_TABLE_NAME not in (row.name, row.owner) and (not_null or row.kind != "con_n")
    }


def _names_006() -> set[tuple[str, ...]]:
    lines = _NAMES_006.read_text().splitlines()
    return {tuple(line.split()) for line in lines if line and not line.startswith("#")}


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
    with engine.begin() as conn:
        conn.execute(
            text(
                f'INSERT INTO "{_SCHEMA}".kelt_atomic_facts '
                "(type, content, content_hash, source, confidence, active, created_at) "
                "VALUES ('assertion', 'fact one', 'h1', 'user', 1.0, true, now())"
            )
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

    def test_downgrade_restores_006_names(self, schema):
        """The downgrade yields exactly the names a schema at 006 has."""
        manager, engine = schema
        _downgrade_to_006(manager, engine)

        assert _version(engine) == "006"
        assert _names(engine) == _names_006()

    def test_upgrade_matches_fresh_schema(self, schema):
        """Upgrading from 006 yields the names of a schema bootstrapped at head, data intact."""
        manager, engine = schema
        fresh = _names(engine, not_null=True)
        _downgrade_to_006(manager, engine)

        status = manager.ensure_schema()

        assert status.current_version == "007"
        assert _names(engine, not_null=True) == fresh
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
        ],
    )
    def test_upgrade_refuses_and_rolls_back(self, schema, setup_sql, error):
        """A name no rule covers, or one too long once prefixed, fails the whole upgrade."""
        manager, engine = schema
        _downgrade_to_006(manager, engine)
        with engine.begin() as conn:
            conn.execute(text(f'SET LOCAL search_path TO "{_SCHEMA}"'))
            conn.execute(text(setup_sql))
        before = _names(engine)

        with pytest.raises(RuntimeError, match=error):
            manager.ensure_schema()

        assert _version(engine) == "006"
        assert _names(engine) == before
