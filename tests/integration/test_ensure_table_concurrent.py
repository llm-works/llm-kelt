# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-kelt Authors

"""Integration test for concurrent ensure_table() first-touch.

Verifies that advisory-lock serialization in ensure_table() handles
concurrent first-touch correctly:

1. N threads racing on ensure_table() for a fresh table all succeed —
   advisory locks serialize the DDL so only one thread creates the table
   while others wait and then find it already exists.
2. The table and HNSW index (for pgvector formats) exist exactly once
   after the race completes.
3. No thread raises an exception.

Reproduces the bug reported in llm-xray's fresh-boot startup, where two
worker processes racing on ensure_table() collided on
pg_type_typname_nsp_index for embeddings_384_f16 (now prevented by
advisory lock).
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, wait
from typing import Any

import pytest
from sqlalchemy import text


@pytest.mark.integration
class TestConcurrentEnsureTable:
    """Concurrent first-touch on a fresh schema must not raise."""

    def _run_race(
        self,
        database: Any,
        test_context: str,
        fmt_value: str,
        threads: int = 8,
    ) -> tuple[str, list[BaseException]]:
        """Spawn ``threads`` workers all racing on ensure_table for a
        fresh table. Returns (table_name, list-of-errors)."""
        from llm_kelt.embedding import Config as EmbeddingConfig
        from llm_kelt.embedding import Factory as EmbeddingFactory
        from llm_kelt.embedding import QuantizationFormat

        prefix = f"race{fmt_value}{test_context[:8]}"
        cfg = EmbeddingConfig(
            context_key="race_test",
            format=QuantizationFormat(fmt_value),
            dimensions=8,
            prefix=prefix,
        )
        factory = EmbeddingFactory()
        table_name = cfg.table_name

        with database.session() as sess:
            sess.connection().execute(text(f"DROP TABLE IF EXISTS {table_name} CASCADE"))
            sess.commit()

        barrier = threading.Barrier(threads)
        errors: list[BaseException] = []
        errors_lock = threading.Lock()

        def worker() -> None:
            # Each thread gets its own StoreClient — StoreClient's own
            # _table_ensured flag is per-instance, so every thread's
            # ensure_table() actually traverses the DDL race path.
            #
            # ``factory.create`` runs before the barrier: any setup failure
            # here (e.g. a metadata race in the model cache) must abort the
            # barrier so the other workers don't hang for 60s waiting on a
            # thread that already died.
            try:
                client = factory.create(database.session, cfg)
            except BaseException as e:  # noqa: BLE001 — surface setup errors
                barrier.abort()
                with errors_lock:
                    errors.append(e)
                return
            try:
                barrier.wait(timeout=60)
                client._store.ensure_table()  # type: ignore[attr-defined]
            except threading.BrokenBarrierError:
                pass  # Another worker failed setup; recorded via its own errors entry.
            except BaseException as e:  # noqa: BLE001 — capture everything
                with errors_lock:
                    errors.append(e)

        with ThreadPoolExecutor(max_workers=threads) as ex:
            futures = [ex.submit(worker) for _ in range(threads)]
            wait(futures)

        return table_name, errors

    def _assert_final_state(self, database: Any, table_name: str, has_index: bool) -> None:
        """After the race, the table exists exactly once and the HNSW
        index (for pgvector formats) also exists exactly once."""
        with database.session() as sess:
            conn = sess.connection()
            row_count = conn.execute(
                text(
                    "SELECT COUNT(*) FROM pg_catalog.pg_class c "
                    "JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace "
                    "WHERE c.relname = :t "
                    "AND n.nspname = ANY(current_schemas(true))"
                ),
                {"t": table_name},
            ).scalar()
            assert row_count == 1, f"expected 1 {table_name} table, found {row_count}"
            if has_index:
                idx_count = conn.execute(
                    text("SELECT COUNT(*) FROM pg_indexes WHERE indexname = :i"),
                    {"i": f"idx_{table_name}_hnsw"},
                ).scalar()
                assert idx_count == 1, f"expected 1 HNSW index, found {idx_count}"

    def _cleanup(self, database: Any, table_name: str) -> None:
        with database.session() as sess:
            sess.connection().execute(text(f"DROP TABLE IF EXISTS {table_name} CASCADE"))
            sess.commit()

    def test_f16_concurrent_first_touch(self, database: Any, test_context: str) -> None:
        """F16 store: 8 threads race on CREATE TABLE + CREATE INDEX
        against a fresh schema. None should raise."""
        table_name, errors = self._run_race(database, test_context, "f16")
        try:
            assert not errors, (
                f"Concurrent ensure_table raised {len(errors)} error(s): "
                f"{[type(e).__name__ + ': ' + str(e)[:200] for e in errors]}"
            )
            self._assert_final_state(database, table_name, has_index=True)
        finally:
            self._cleanup(database, table_name)

    def test_f32_concurrent_first_touch(self, database: Any, test_context: str) -> None:
        """Same race, F32 store — the other pgvector path with HNSW."""
        table_name, errors = self._run_race(database, test_context, "f32")
        try:
            assert not errors, (
                f"Concurrent ensure_table raised {len(errors)} error(s): "
                f"{[type(e).__name__ + ': ' + str(e)[:200] for e in errors]}"
            )
            self._assert_final_state(database, table_name, has_index=True)
        finally:
            self._cleanup(database, table_name)

    def test_i8_concurrent_first_touch(self, database: Any, test_context: str) -> None:
        """I8 store has no HNSW index but the CREATE TABLE race is the same."""
        table_name, errors = self._run_race(database, test_context, "i8")
        try:
            assert not errors, (
                f"Concurrent ensure_table raised {len(errors)} error(s): "
                f"{[type(e).__name__ + ': ' + str(e)[:200] for e in errors]}"
            )
            self._assert_final_state(database, table_name, has_index=False)
        finally:
            self._cleanup(database, table_name)
