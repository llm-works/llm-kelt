"""Unit tests for the schema-aware / race-safe ensure_table path.

Guards two regressions:

1. table_exists() must resolve the target namespace by name, not by
   pg_table_is_visible() — the latter is search_path-dependent, so a
   caller that manages search_path per-statement (rather than at session
   setup) sees a false negative and re-issues CREATE TABLE. The CREATE
   then lands in the same schema as the existing table and collides on
   pg_type_typname_nsp_index for the auto-generated composite type.

2. ensure_table() must swallow duplicate-name errors on both the CREATE
   TABLE and CREATE INDEX paths. Even with a schema-aware existence
   check, two workers racing on first-touch each observe the empty
   schema at T1, both fire CREATE at T2, and the second raises
   IntegrityError on the composite type's pg_type_typname_nsp_index (or
   ProgrammingError "relation already exists"). The savepoint absorbs
   the failure without aborting the outer transaction.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

from sqlalchemy.exc import IntegrityError, ProgrammingError

from llm_kelt.embedding.store import base
from llm_kelt.embedding.store.f16 import Float16Store
from llm_kelt.embedding.store.f32 import Float32Store
from llm_kelt.embedding.store.i4 import Int4Store
from llm_kelt.embedding.store.i8 import Int8Store


class _CapturingConn:
    """Minimal SQLAlchemy connection stub that records the SQL text and
    parameters passed to execute() and returns a canned scalar."""

    def __init__(self, scalar_value: Any) -> None:
        self._scalar = scalar_value
        self.last_sql: str = ""
        self.last_params: dict[str, Any] = {}

    def execute(self, textclause: Any) -> Any:  # noqa: ANN401
        self.last_sql = str(textclause)
        self.last_params = dict(textclause.compile().params)
        return SimpleNamespace(scalar=lambda: self._scalar)


class TestTableExistsSchema:
    """table_exists() SQL construction is schema-aware."""

    def test_with_schema_filters_by_nspname(self) -> None:
        conn = _CapturingConn(scalar_value=1)

        assert base.table_exists(conn, "embeddings_384_f16", schema="prod") is True

        assert "n.nspname = :schema" in conn.last_sql
        assert "pg_table_is_visible" not in conn.last_sql
        assert conn.last_params["schema"] == "prod"
        assert conn.last_params["table_name"] == "embeddings_384_f16"

    def test_without_schema_uses_current_schemas(self) -> None:
        conn = _CapturingConn(scalar_value=None)

        assert base.table_exists(conn, "embeddings_384_f16") is False

        assert "current_schemas(true)" in conn.last_sql
        assert "pg_table_is_visible" not in conn.last_sql
        assert conn.last_params["table_name"] == "embeddings_384_f16"


class TestIndexExistsSchema:
    """index_exists() SQL construction is schema-aware."""

    def test_with_schema_filters_by_schemaname(self) -> None:
        conn = _CapturingConn(scalar_value=1)

        assert base.index_exists(conn, "idx_embeddings_384_f16_hnsw", schema="prod") is True

        assert "schemaname = :schema" in conn.last_sql
        assert conn.last_params["schema"] == "prod"
        assert conn.last_params["idx_name"] == "idx_embeddings_384_f16_hnsw"

    def test_without_schema_matches_any_namespace(self) -> None:
        conn = _CapturingConn(scalar_value=None)

        assert base.index_exists(conn, "idx_embeddings_384_f16_hnsw") is False

        assert "schemaname" not in conn.last_sql
        assert conn.last_params["idx_name"] == "idx_embeddings_384_f16_hnsw"


class _Nested:
    """Stand-in for session.begin_nested()'s SessionTransaction context.

    Real SAVEPOINT rollback happens inside the SQLAlchemy Session on
    __exit__; the mock only needs to propagate the exception so the
    surrounding try/except in ensure_table() can catch it.
    """

    def __enter__(self) -> _Nested:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        return False


class _SessionCM:
    def __init__(self, session: Any) -> None:
        self._session = session

    def __enter__(self) -> Any:
        return self._session

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        return False


def _make_fake_session() -> tuple[Any, MagicMock]:
    """Build a session that records commit() and yields a fresh Nested."""
    conn = MagicMock(name="conn")
    session = MagicMock(name="session")
    session.connection.return_value = conn
    session.begin_nested.side_effect = _Nested
    return session, conn


def _fake_table(create_side_effect: Exception | None = None) -> SimpleNamespace:
    """SQLAlchemy Table stand-in with .schema, .create(), .drop()."""
    create = MagicMock(
        name="table.create",
        side_effect=create_side_effect if create_side_effect is not None else None,
    )
    return SimpleNamespace(schema=None, create=create)


class _FakeModel:
    """Lightweight model stub. Uses a class rather than SimpleNamespace
    so store.table_name can read __tablename__ via attribute access."""

    def __init__(self, table: Any, tablename: str = "embeddings_test") -> None:
        self.__table__ = table
        self.__tablename__ = tablename


class TestEnsureTableSwallowsCreateRace:
    """ensure_table() must not raise when CREATE loses the race."""

    def _dup_key_error(self) -> IntegrityError:
        return IntegrityError(
            "CREATE TABLE embeddings_test (...)",
            {},
            Exception("duplicate key value violates unique constraint pg_type_typname_nsp_index"),
        )

    def test_f16_create_table_race_is_swallowed(self, monkeypatch: Any) -> None:
        session, _ = _make_fake_session()
        table = _fake_table(create_side_effect=self._dup_key_error())
        model = _FakeModel(table)
        store = Float16Store(
            session_factory=lambda: _SessionCM(session), dimensions=384, model=model
        )
        monkeypatch.setattr(
            "llm_kelt.embedding.store.f16.table_exists",
            lambda conn, name, schema=None: False,
        )
        monkeypatch.setattr(
            "llm_kelt.embedding.store.f16.index_exists",
            lambda conn, name, schema=None: True,
        )

        store.ensure_table()

        assert store._table_ensured is True
        session.commit.assert_called_once()
        assert table.create.call_count == 1

    def test_f32_create_table_race_is_swallowed(self, monkeypatch: Any) -> None:
        session, _ = _make_fake_session()
        table = _fake_table(create_side_effect=self._dup_key_error())
        model = _FakeModel(table)
        store = Float32Store(
            session_factory=lambda: _SessionCM(session), dimensions=384, model=model
        )
        monkeypatch.setattr(
            "llm_kelt.embedding.store.f32.table_exists",
            lambda conn, name, schema=None: False,
        )
        monkeypatch.setattr(
            "llm_kelt.embedding.store.f32.index_exists",
            lambda conn, name, schema=None: True,
        )

        store.ensure_table()

        assert store._table_ensured is True
        session.commit.assert_called_once()

    def test_i8_create_table_race_is_swallowed(self, monkeypatch: Any) -> None:
        session, _ = _make_fake_session()
        table = _fake_table(create_side_effect=self._dup_key_error())
        model = _FakeModel(table)
        store = Int8Store(session_factory=lambda: _SessionCM(session), dimensions=384, model=model)
        monkeypatch.setattr(
            "llm_kelt.embedding.store.i8.table_exists",
            lambda conn, name, schema=None: False,
        )

        store.ensure_table()

        assert store._table_ensured is True
        session.commit.assert_called_once()

    def test_i4_create_table_race_is_swallowed(self, monkeypatch: Any) -> None:
        session, _ = _make_fake_session()
        table = _fake_table(create_side_effect=self._dup_key_error())
        model = _FakeModel(table)
        store = Int4Store(session_factory=lambda: _SessionCM(session), dimensions=384, model=model)
        monkeypatch.setattr(
            "llm_kelt.embedding.store.i4.table_exists",
            lambda conn, name, schema=None: False,
        )

        store.ensure_table()

        assert store._table_ensured is True
        session.commit.assert_called_once()

    def test_f16_relation_already_exists_is_swallowed(self, monkeypatch: Any) -> None:
        """ProgrammingError 'relation already exists' path also survives."""
        session, _ = _make_fake_session()
        table = _fake_table(
            create_side_effect=ProgrammingError(
                "CREATE TABLE ...",
                {},
                Exception('relation "embeddings_test" already exists'),
            )
        )
        model = _FakeModel(table)
        store = Float16Store(
            session_factory=lambda: _SessionCM(session), dimensions=384, model=model
        )
        monkeypatch.setattr(
            "llm_kelt.embedding.store.f16.table_exists",
            lambda conn, name, schema=None: False,
        )
        monkeypatch.setattr(
            "llm_kelt.embedding.store.f16.index_exists",
            lambda conn, name, schema=None: True,
        )

        store.ensure_table()

        assert store._table_ensured is True
        session.commit.assert_called_once()


class TestEnsureTableSwallowsIndexRace:
    """The HNSW index CREATE has the same race window; verify it too."""

    def test_f16_create_index_race_is_swallowed(self, monkeypatch: Any) -> None:
        session, conn = _make_fake_session()
        conn.execute.side_effect = IntegrityError(
            "CREATE INDEX ...", {}, Exception("duplicate key")
        )
        table = _fake_table()
        model = _FakeModel(table)
        store = Float16Store(
            session_factory=lambda: _SessionCM(session), dimensions=384, model=model
        )
        monkeypatch.setattr(
            "llm_kelt.embedding.store.f16.table_exists",
            lambda conn, name, schema=None: True,
        )
        monkeypatch.setattr(
            "llm_kelt.embedding.store.f16.index_exists",
            lambda conn, name, schema=None: False,
        )

        store.ensure_table()

        assert store._table_ensured is True
        session.commit.assert_called_once()
        assert conn.execute.call_count == 1

    def test_f32_create_index_race_is_swallowed(self, monkeypatch: Any) -> None:
        session, conn = _make_fake_session()
        conn.execute.side_effect = IntegrityError(
            "CREATE INDEX ...", {}, Exception("duplicate key")
        )
        table = _fake_table()
        model = _FakeModel(table)
        store = Float32Store(
            session_factory=lambda: _SessionCM(session), dimensions=384, model=model
        )
        monkeypatch.setattr(
            "llm_kelt.embedding.store.f32.table_exists",
            lambda conn, name, schema=None: True,
        )
        monkeypatch.setattr(
            "llm_kelt.embedding.store.f32.index_exists",
            lambda conn, name, schema=None: False,
        )

        store.ensure_table()

        assert store._table_ensured is True
        session.commit.assert_called_once()


class TestEnsureTablePassesSchemaToChecks:
    """When the model's __table__.schema is set, the existence checks
    receive it instead of falling back to search_path resolution."""

    def test_f16_forwards_schema(self, monkeypatch: Any) -> None:
        session, _ = _make_fake_session()
        table = _fake_table()
        table.schema = "prod"
        model = _FakeModel(table)
        store = Float16Store(
            session_factory=lambda: _SessionCM(session), dimensions=384, model=model
        )
        seen_schemas: list[str | None] = []

        def _table_exists(conn: Any, name: str, schema: str | None = None) -> bool:
            seen_schemas.append(schema)
            return True

        def _index_exists(conn: Any, name: str, schema: str | None = None) -> bool:
            seen_schemas.append(schema)
            return True

        monkeypatch.setattr("llm_kelt.embedding.store.f16.table_exists", _table_exists)
        monkeypatch.setattr("llm_kelt.embedding.store.f16.index_exists", _index_exists)

        store.ensure_table()

        assert seen_schemas == ["prod", "prod"]
