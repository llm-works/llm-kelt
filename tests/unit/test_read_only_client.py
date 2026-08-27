# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-kelt Authors

"""Tests for the read-only Client construction path.

Ticket: lightweight consumers that only issue SELECTs (no vectors, no
embeddings) must be able to construct a Client without pulling
pgvector/numpy at import time.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from unittest.mock import MagicMock, Mock, patch

import pytest

from llm_kelt import ClientContext, SchemaMode
from llm_kelt.client import Client
from llm_kelt.core.errors import SchemaVersionError


@pytest.fixture
def mock_database():
    db = Mock()
    db.session = MagicMock()
    db.engine = Mock()
    db.schema = None
    return db


@pytest.fixture
def mock_logger():
    return Mock()


class TestReadOnlyConstruction:
    """schema_mode=SKIP + no embedder should not touch alembic or pgvector."""

    def test_skip_mode_does_not_instantiate_schema_manager(self, mock_logger, mock_database):
        context = ClientContext(context_key="reader")
        with patch("llm_kelt.client.SchemaManager") as mock_mgr:
            Client(
                lg=mock_logger,
                database=mock_database,
                context=context,
                schema_mode=SchemaMode.SKIP,
            )
        mock_mgr.assert_not_called()
        mock_database.ensure_schema.assert_not_called()

    def test_no_embedder_leaves_factory_unset(self, mock_logger, mock_database):
        context = ClientContext(context_key="reader")
        with patch("llm_kelt.client.EmbeddingFactory") as mock_factory:
            client = Client(
                lg=mock_logger,
                database=mock_database,
                context=context,
                schema_mode=SchemaMode.SKIP,
            )
        mock_factory.assert_not_called()
        assert client._embedding_factory is None
        assert client._embeddings is None

    def test_embeddings_property_raises_friendly_error(self, mock_logger, mock_database):
        context = ClientContext(context_key="reader")
        client = Client(
            lg=mock_logger,
            database=mock_database,
            context=context,
            schema_mode=SchemaMode.SKIP,
        )
        with pytest.raises(RuntimeError, match="Embeddings not configured"):
            _ = client.embeddings

    def test_atomic_surface_available_without_embeddings(self, mock_logger, mock_database):
        context = ClientContext(context_key="reader")
        client = Client(
            lg=mock_logger,
            database=mock_database,
            context=context,
            schema_mode=SchemaMode.SKIP,
        )
        # Just resolving the atomic protocol and its assertions client must
        # not require an embedding factory. The read path (get/get_many)
        # never touches embeddings.
        assert client.atomic is not None
        assert client.atomic.assertions is not None


class TestVerifyMode:
    """schema_mode=VERIFY still walks alembic and validates the head."""

    def test_verify_raises_when_schema_not_current(self, mock_logger, mock_database):
        context = ClientContext(context_key="reader")
        with patch("llm_kelt.client.SchemaManager") as mock_mgr_cls:
            manager = Mock()
            status = Mock()
            status.state = Mock(value="missing")
            status.current_version = None
            status.head_version = "007"
            manager.get_status.return_value = status
            mock_mgr_cls.return_value = manager

            with pytest.raises(SchemaVersionError):
                Client(
                    lg=mock_logger,
                    database=mock_database,
                    context=context,
                    schema_mode=SchemaMode.VERIFY,
                )


# ---------------------------------------------------------------------------
# Subprocess-level test: prove the ticket's reproducer.
#
# Blocks numpy at the import loader and instantiates a read-only Client. If
# construction succeeds, the pgvector→numpy import chain has genuinely been
# removed from the read-only path. Uses a Database stub so we don't need a
# real Postgres to run the check.
# ---------------------------------------------------------------------------

_SUBPROCESS_SCRIPT = textwrap.dedent(
    """
    import sys

    class _BlockNumpy:
        def find_spec(self, name, path=None, target=None):
            if name == "numpy" or name.startswith("numpy."):
                raise ModuleNotFoundError(f"blocked: {name}")
            return None

    sys.meta_path.insert(0, _BlockNumpy())

    from unittest.mock import MagicMock, Mock

    from llm_kelt import Client, ClientContext, SchemaMode

    db = Mock()
    db.session = MagicMock()
    db.engine = Mock()
    db.schema = None

    ctx = ClientContext(context_key="reader")
    client = Client(
        lg=Mock(),
        database=db,
        context=ctx,
        schema_mode=SchemaMode.SKIP,
    )

    assert "numpy" not in sys.modules, sorted(m for m in sys.modules if "numpy" in m)
    assert "pgvector" not in sys.modules
    print("OK")
    """
).strip()


def test_read_only_client_does_not_import_numpy(tmp_path):
    """Ticket reproducer: block numpy at import, construct a read-only Client.

    Runs in a subprocess so import state stays isolated from the pytest
    process (which has already imported numpy for other tests).
    """
    script = tmp_path / "reader.py"
    script.write_text(_SUBPROCESS_SCRIPT)

    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, (
        f"read-only Client pulled numpy at import.\n"
        f"stdout: {result.stdout}\n"
        f"stderr: {result.stderr}"
    )
    assert "OK" in result.stdout
