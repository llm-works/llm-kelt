"""Tests for schema management and migrations."""

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest
from sqlalchemy import text

from llm_kelt.core.errors import SchemaVersionError
from llm_kelt.core.schema import SchemaManager, SchemaState


@pytest.fixture(autouse=True)
def ensure_alembic_version(logger, database):
    """Ensure alembic_version table exists before running schema tests.

    This is needed because the tests manipulate the alembic_version table directly,
    and when running in parallel, the table might not exist if another test hasn't
    called ensure_schema() first.
    """
    manager = SchemaManager(logger, database.engine, schema_name=database.schema)
    manager.ensure_schema()
    yield


# Group all schema tests to run on the same worker (they manipulate alembic_version)
@pytest.mark.xdist_group("schema")
class TestSchemaManager:
    """Test SchemaManager functionality."""

    def test_get_status_current(self, logger, database):
        """Test that get_status returns CURRENT after ensure_schema."""
        manager = SchemaManager(logger, database.engine, schema_name=database.schema)

        # First ensure schema is initialized (stamps alembic_version)
        manager.ensure_schema()

        status = manager.get_status()

        assert status.state == SchemaState.CURRENT
        assert status.current_version is not None
        assert status.head_version is not None
        assert status.current_version == status.head_version

    def test_ensure_schema_idempotent(self, logger, database):
        """Test that ensure_schema is idempotent - multiple calls succeed."""
        manager = SchemaManager(logger, database.engine, schema_name=database.schema)

        # Call multiple times
        status1 = manager.ensure_schema()
        status2 = manager.ensure_schema()
        status3 = manager.ensure_schema()

        assert status1.state == SchemaState.CURRENT
        assert status2.state == SchemaState.CURRENT
        assert status3.state == SchemaState.CURRENT
        assert status1.current_version == status2.current_version == status3.current_version

    def test_ensure_schema_concurrent(self, logger, database):
        """Test that concurrent ensure_schema calls are safe.

        Note: This tests thread-safety within a single process. Cross-process safety
        relies on PostgreSQL advisory locks with a fixed lock key (not Python's hash()).
        Multi-process testing would require subprocess spawning which is more complex
        and typically done in integration/e2e test suites.
        """
        results = []
        errors = []
        lock = threading.Lock()

        def call_ensure_schema():
            try:
                manager = SchemaManager(logger, database.engine, schema_name=database.schema)
                status = manager.ensure_schema()
                with lock:
                    results.append(status)
            except Exception as e:
                with lock:
                    errors.append(e)

        # Run 5 concurrent threads
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(call_ensure_schema) for _ in range(5)]
            for future in as_completed(futures):
                future.result()  # Raise any exceptions

        assert len(errors) == 0, f"Concurrent errors: {errors}"
        assert len(results) == 5
        assert all(r.state == SchemaState.CURRENT for r in results)

    def test_downgrade_protection(self, logger, database):
        """Test that SchemaVersionError is raised for unknown (future) versions."""
        manager = SchemaManager(logger, database.engine, schema_name=database.schema)

        # Insert a fake future version
        with database.engine.connect() as conn:
            conn.execute(text("DELETE FROM alembic_version"))
            conn.execute(
                text("INSERT INTO alembic_version (version_num) VALUES ('9999_future_version')")
            )
            conn.commit()

        try:
            status = manager.get_status()
            assert status.state == SchemaState.TOO_NEW

            with pytest.raises(SchemaVersionError, match="newer than"):
                manager.ensure_schema()
        finally:
            # Restore correct version
            head = manager._get_head_version()
            with database.engine.connect() as conn:
                conn.execute(text("DELETE FROM alembic_version"))
                conn.execute(
                    text("INSERT INTO alembic_version (version_num) VALUES (:rev)"),
                    {"rev": head},
                )
                conn.commit()

    def test_get_status_missing(self, logger, database):
        """Test that get_status returns MISSING when alembic_version is empty."""
        manager = SchemaManager(logger, database.engine, schema_name=database.schema)
        head = manager._get_head_version()

        # Clear alembic_version
        with database.engine.connect() as conn:
            conn.execute(text("DELETE FROM alembic_version"))
            conn.commit()

        try:
            status = manager.get_status()
            assert status.state == SchemaState.MISSING
            assert status.current_version is None
            assert status.head_version == head
        finally:
            # Restore
            with database.engine.connect() as conn:
                conn.execute(
                    text("INSERT INTO alembic_version (version_num) VALUES (:rev)"),
                    {"rev": head},
                )
                conn.commit()

    def test_lock_timeout_non_blocking(self, logger, database):
        """Test that non-blocking lock fails fast when lock is held."""
        from llm_kelt.core.schema import _ADVISORY_LOCK_KEY

        manager = SchemaManager(logger, database.engine, schema_name=database.schema)

        # Acquire lock manually using the same fixed key as SchemaManager
        with database.engine.connect() as holder_conn:
            holder_conn.execute(text(f"SELECT pg_advisory_lock({_ADVISORY_LOCK_KEY})"))

            try:
                # Non-blocking attempt on a different connection should fail
                with database.engine.connect() as test_conn:
                    acquired = manager._acquire_lock(test_conn, wait=False, timeout_seconds=1.0)
                    assert acquired is False
            finally:
                holder_conn.execute(text(f"SELECT pg_advisory_unlock({_ADVISORY_LOCK_KEY})"))
                holder_conn.commit()


@pytest.mark.xdist_group("schema")
class TestEnsureSchemaHelper:
    """Test the top-level llm_kelt.ensure_schema() helper."""

    def test_public_api_idempotent(self, logger, pg_with_tables):
        """ensure_schema is exported from llm_kelt and is idempotent."""
        import llm_kelt

        status1 = llm_kelt.ensure_schema(logger, pg_with_tables)
        status2 = llm_kelt.ensure_schema(logger, pg_with_tables)

        # Return type and enum come from the public API surface — no need to
        # reach into llm_kelt.core.* to inspect the result.
        assert isinstance(status1, llm_kelt.SchemaStatus)
        assert status1.state == llm_kelt.SchemaState.CURRENT
        assert status2.state == llm_kelt.SchemaState.CURRENT
        assert status1.current_version == status2.current_version
        assert status1.current_version == status1.head_version

    def test_schema_name_override(self, logger, pg_with_tables):
        """Explicit schema_name parameter overrides pg.schema.

        Passes a schema name distinct from `pg_with_tables.schema` and verifies
        that migrations land in the override schema (not the PG default).
        """
        import llm_kelt

        override_schema = "ensure_schema_override_test"
        assert pg_with_tables.schema != override_schema, (
            "fixture precondition: override schema must differ from pg.schema"
        )

        try:
            # SchemaManager runs migrations with search_path=override_schema but
            # does not CREATE the schema itself (ensure_pg_schema only creates
            # pg.schema). Pre-create it so migrations have a target.
            with pg_with_tables.engine.connect() as conn:
                conn.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{override_schema}"'))
                conn.commit()

            status = llm_kelt.ensure_schema(logger, pg_with_tables, schema_name=override_schema)
            assert status.state == llm_kelt.SchemaState.CURRENT

            # Verify tables actually landed in the override schema, not pg.schema.
            with pg_with_tables.engine.connect() as conn:
                result = conn.execute(
                    text(
                        "SELECT COUNT(*) FROM information_schema.tables "
                        "WHERE table_schema = :schema AND table_name = 'alembic_version'"
                    ),
                    {"schema": override_schema},
                ).scalar()
                assert result == 1, (
                    f"alembic_version should exist in override schema '{override_schema}'"
                )
        finally:
            with pg_with_tables.engine.connect() as conn:
                conn.execute(text(f'DROP SCHEMA IF EXISTS "{override_schema}" CASCADE'))
                conn.commit()

    def test_empty_schema_name_rejected(self, logger, pg_with_tables):
        """Empty string schema_name is rejected explicitly (not silently ignored)."""
        import llm_kelt

        with pytest.raises(ValueError, match="cannot be empty"):
            llm_kelt.ensure_schema(logger, pg_with_tables, schema_name="")
