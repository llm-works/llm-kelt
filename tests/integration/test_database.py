# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-kelt Authors

"""Database connection pool lifecycle against Postgres."""

from sqlalchemy import text


class TestDispose:
    """Database.dispose() closes pooled connections without ending the Database."""

    def test_dispose_closes_pooled_connections(self, database):
        """Connections returned to the pool are closed."""
        with database.session() as session:
            session.execute(text("SELECT 1"))
        assert database.engine.pool.checkedin() > 0

        database.dispose()

        assert database.engine.pool.checkedin() == 0

    def test_sessions_reconnect_after_dispose(self, database):
        """A session opened after dispose() gets a new connection."""
        database.dispose()

        with database.session() as session:
            assert session.execute(text("SELECT 1")).scalar() == 1
