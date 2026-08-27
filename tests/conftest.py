# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-kelt Authors

"""Pytest fixtures for Kelt framework tests."""

import os
import socket
import sys
from pathlib import Path
from urllib.parse import urlparse

import pytest
from appinfra.config import Config
from appinfra.log import LogConfig, LoggerFactory

from llm_kelt.client import Client
from llm_kelt.core.database import Database
from llm_kelt.core.models import Base

# Import atomic memory models so they're registered with Base for migrations
from llm_kelt.memory.atomic import models as atomic_models  # noqa: F401

# Import KG models so they're registered with Base for migrations
from llm_kelt.memory.kg import models as kg_models  # noqa: F401

# Import training models so they're registered with Base for migrations
from llm_kelt.training import dpo as training_dpo  # noqa: F401
from llm_kelt.training import sft as training_sft  # noqa: F401

# Enable appinfra's schema isolation fixtures for parallel test execution
pytest_plugins = ["appinfra.db.pg.testing"]


@pytest.fixture(scope="session")
def pg_test_schema(worker_id: str) -> str:
    """
    Generate a unique schema name per pytest process.

    Overrides appinfra.db.pg.testing.pg_test_schema to include PID, preventing
    collisions when multiple pytest processes run simultaneously (e.g., make check
    running test.integration and test.coverage in parallel).

    Schema naming:
    - xdist worker: test_gw0_12345, test_gw1_12345, etc.
    - non-xdist: test_master_12345

    Dependencies:
    - worker_id: provided by pytest-xdist (defaults to "master" when not using xdist)
    - Cleanup: handled by appinfra's pg_migrate_factory (DROP SCHEMA in finally block)

    Note: If appinfra's pg_test_schema signature changes, this override must be updated.
    """
    pid = os.getpid()
    if worker_id == "master":
        return f"test_master_{pid}"
    return f"test_{worker_id}_{pid}"


@pytest.hookimpl(tryfirst=True)
def pytest_cmdline_main(config):
    """Force sequential execution for e2e tests (GPU can't be shared).

    Uses pytest_cmdline_main with tryfirst=True because xdist decides to spawn
    workers in its pytest_cmdline_main hook. We need to set numprocesses=0
    BEFORE xdist's hook runs.
    """
    # Check if running e2e tests
    markexpr = config.getoption("-m", default="")
    if "e2e" in markexpr:
        # Force sequential execution by setting numprocesses to 0
        if hasattr(config.option, "numprocesses") and config.option.numprocesses:
            original = config.option.numprocesses
            config.option.numprocesses = 0
            print(f"\n*** E2E detected: forcing sequential (was -n {original}) ***\n")


def pytest_configure(config):
    """Probe Postgres once per session and stash the result."""
    endpoint = _resolve_pg_endpoint()
    if endpoint is None:
        return
    host, port = endpoint
    available = _is_server_available(host, port)
    config.stash[_PG_STATUS_KEY] = {"host": host, "port": port, "available": available}
    if not available:
        print(
            f"PG probe: {host}:{port} unreachable; PG-dependent tests will skip "
            f"with reason '{_PG_SKIP_REASON}'",
            file=sys.stderr,
        )


# Find project root and config paths
PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "etc" / "llm-kelt.yaml"


def _get_config_path() -> Path:
    """Get config path from env var or default."""
    config_file = os.environ.get("KELT_TEST_CONFIG_FILE")
    if config_file:
        return Path(config_file)
    return DEFAULT_CONFIG_PATH


def _is_server_available(host: str, port: int, timeout: float = 1.0) -> bool:
    """Check if a server is accepting connections."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (OSError, ConnectionRefusedError):
        return False


def _get_llm_server_address() -> tuple[str, int] | None:
    """Extract LLM server host/port from config."""
    config_path = _get_config_path()
    if not config_path.exists():
        return None

    config = Config(str(config_path))
    llm_config = getattr(config, "llm", None)
    if not llm_config:
        return None

    # Get the local backend URL
    backends = getattr(llm_config, "backends", None)
    if not backends:
        return None

    local_backend = getattr(backends, "local", None)
    if not local_backend:
        return None

    base_url = getattr(local_backend, "base_url", None)
    if not base_url:
        return None

    parsed = urlparse(base_url)
    host = parsed.hostname or "localhost"
    port = parsed.port or 8000
    return (host, port)


def pytest_collection_modifyitems(config, items):
    """Auto-apply markers based on test directory, then deselect PG-dependent
    tests when Postgres is unreachable.

    Deselection (vs pytest.skip) keeps the failure-step output clean: skipped
    tests appear in pytest's `-rs` summary as `SKIPPED [1] ...` lines, which
    check.sh dumps when a step fails. Deselected tests just aren't in the run,
    so a downstream failure (e.g. coverage threshold) shows the actual cause,
    not 200+ lines of PG noise.
    """
    for item in items:
        test_path = Path(item.fspath)
        if "unit" in test_path.parts:
            item.add_marker(pytest.mark.unit)
        elif "integration" in test_path.parts:
            item.add_marker(pytest.mark.integration)
        elif "e2e" in test_path.parts:
            item.add_marker(pytest.mark.e2e)

    status = config.stash.get(_PG_STATUS_KEY, None)
    if status is None or status["available"]:
        return
    keep: list = []
    dropped: list = []
    for item in items:
        if _PG_FIXTURE_GATE in item.fixturenames:
            dropped.append(item)
        else:
            keep.append(item)
    if dropped:
        config.hook.pytest_deselected(items=dropped)
        items[:] = keep
        print(
            f"PG probe: {status['host']}:{status['port']} unreachable; "
            f"deselected {len(dropped)} PG-dependent tests",
            file=sys.stderr,
        )


# Cache server availability check (only check once per session)
_llm_server_available: bool | None = None
_llm_server_address: tuple[str, int] | None = None


def pytest_runtest_setup(item):
    """Skip LLM-marked tests if the LLM server is not available."""
    global _llm_server_available, _llm_server_address

    # Only check tests marked with @pytest.mark.llm
    if not any(mark.name == "llm" for mark in item.iter_markers()):
        return

    # Check server availability (cached for session)
    if _llm_server_available is None:
        _llm_server_address = _get_llm_server_address()
        if _llm_server_address:
            host, port = _llm_server_address
            _llm_server_available = _is_server_available(host, port)
        else:
            _llm_server_available = False

    if not _llm_server_available:
        addr = (
            f"{_llm_server_address[0]}:{_llm_server_address[1]}"
            if _llm_server_address
            else "unknown"
        )
        pytest.skip(f"LLM server not available at {addr}")


@pytest.fixture(scope="session")
def config():
    """Load test configuration from KELT_TEST_CONFIG_FILE or default."""
    return Config(str(_get_config_path()))


@pytest.fixture(scope="session")
def llm_config(config):
    """Get LLM configuration dict for LLMClient.from_config()."""
    return config.llm.to_dict()


@pytest.fixture
def llm_client(llm_config, logger):
    """Create LLM client from test config."""
    from llm_infer.client import Factory as LLMClientFactory

    factory = LLMClientFactory(logger)
    return factory.from_config(llm_config)


@pytest.fixture(scope="session")
def logger():
    """Create test logger."""
    log_config = LogConfig.from_params(level="debug")
    return LoggerFactory.create_root(log_config)


@pytest.fixture(scope="session")
def lg(logger):
    """Alias for logger fixture (matches parameter naming convention)."""
    return logger


# Bottleneck fixture every PG-dependent test transitively pulls in (via
# `database`, `kelt_client`, `pg_with_tables`, …). Used by
# pytest_collection_modifyitems to identify those tests and deselect them
# en masse when the Postgres probe fails at session start.
_PG_FIXTURE_GATE = "pg_test_config"
_PG_SKIP_REASON = "pg-unavailable"
_PG_STATUS_KEY: pytest.StashKey[dict] = pytest.StashKey()


def _has_module(name: str) -> bool:
    """True if ``name`` can be imported in the current environment."""
    import importlib.util

    return importlib.util.find_spec(name) is not None


# Ignore test files whose optional deps are missing. The contained tests use
# inline ``pytest.importorskip``; running them surfaces per-test SKIPPED lines
# that clutter the failure dump when an upstream step (e.g. coverage) reports
# the log tail. Skipping at collection means pytest never sees the file.
collect_ignore: list[str] = []
if not _has_module("trl") or not _has_module("peft"):
    collect_ignore.append("unit/test_dpo_reference_modes.py")


def _db_url_address(url: str) -> tuple[str, int] | None:
    parsed = urlparse(url)
    if not parsed.hostname:
        return None
    return parsed.hostname, parsed.port or 5432


def _resolve_pg_endpoint() -> tuple[str, int] | None:
    """Pick the host:port the test suite will try to connect to.

    Order: DATABASE_URL → dbs.unittest in config → INFRA_PGSERVER_HOST/PORT
    env overrides → None. Mirrors the override paths used downstream in
    pg_test_config and what appinfra-style probes accept.
    """
    database_url = os.environ.get("DATABASE_URL")
    if database_url:
        return _db_url_address(database_url)
    config_path = _get_config_path()
    if config_path.exists():
        try:
            cfg = Config(str(config_path))
            url = cfg.get("dbs.unittest.url")
            if url:
                return _db_url_address(str(url))
        except Exception:  # noqa: BLE001 — config parse failure falls through to env
            pass
    host = os.environ.get("INFRA_PGSERVER_HOST")
    port_str = os.environ.get("INFRA_PGSERVER_PORT")
    if host and port_str:
        try:
            return host, int(port_str)
        except ValueError:
            return None
    return None


@pytest.fixture(scope="session")
def pg_test_config(config):
    """Provide database config to appinfra's schema isolation fixtures.

    Checks for DATABASE_URL environment variable first (used in CI),
    otherwise falls back to config file. When the Postgres probe fails at
    session start, pytest_collection_modifyitems has already deselected
    every test that pulls this fixture in, so the unreachable-PG case
    never reaches us here.
    """
    database_url = os.environ.get("DATABASE_URL")
    if database_url:
        return {
            "url": database_url,
            "create_db": True,
            "readonly": False,
            "pool_pre_ping": True,
            "extensions": ["vector"],
        }

    db_cfg = config.dbs.get("unittest")
    if db_cfg is None:
        pytest.skip("Database config 'dbs.unittest' not found in etc/infra.yaml")
    return db_cfg


@pytest.fixture(scope="session")
def pg_with_tables(pg_migrate_factory):
    """PG instance with schema isolation and migrations applied."""
    with pg_migrate_factory(Base, extensions=["vector"]) as pg:
        yield pg


@pytest.fixture(scope="session")
def database(logger, pg_with_tables):
    """Create Database wrapper from PG with migrations applied."""
    return Database(logger, pg_with_tables)


@pytest.fixture
def test_context(request):
    """Return a unique test context key per test.

    Uses the test node ID to ensure isolation between parallel tests.
    This prevents flakiness when tests run concurrently with xdist.
    """
    from hashlib import md5

    # Include test name for uniqueness across parallel workers
    test_id = request.node.nodeid.encode()
    return md5(test_id).hexdigest()


@pytest.fixture
def kelt_client(logger, database, test_context):
    """Create Client for testing, scoped to test context.

    Uses 3-dimensional embeddings for test simplicity (matching test vectors).
    """
    from llm_kelt import ClientContext
    from llm_kelt.embedding import Config as EmbeddingConfig
    from llm_kelt.embedding import Factory as EmbeddingFactory
    from llm_kelt.embedding import QuantizationFormat

    # Use small dimensions for tests (3-element vectors are common in test data)
    embedding_config = EmbeddingConfig(
        context_key=test_context or "_test",
        format=QuantizationFormat.F32,  # F32 avoids HalfVector issues
        dimensions=3,
    )
    factory = EmbeddingFactory()
    embeddings = factory.create(database.session, embedding_config)

    context = ClientContext(context_key=test_context, schema_name=None)
    return Client(lg=logger, database=database, context=context, embeddings=embeddings)


@pytest.fixture
def clean_tables(database, test_context):
    """Clean all tables before each test."""
    from sqlalchemy import inspect, text
    from sqlalchemy.exc import ProgrammingError

    with database.session() as session:
        inspector = inspect(session.bind)
        existing_tables = set(inspector.get_table_names())

        # Delete in reverse order to respect foreign keys
        for table in reversed(Base.metadata.sorted_tables):
            # Skip dynamically-created tables that may not exist
            if table.name not in existing_tables:
                continue
            try:
                session.execute(table.delete())
            except ProgrammingError:
                # Table doesn't exist yet (dynamic embedding tables)
                session.rollback()

        # Also clean dynamic embedding tables (not in Base.metadata)
        for table_name in existing_tables:
            if table_name.startswith("embeddings_"):
                try:
                    session.execute(text(f"DELETE FROM {table_name}"))
                except ProgrammingError:
                    session.rollback()
    yield


@pytest.fixture
def sample_content(kelt_client, clean_tables):
    """Create sample content for testing."""
    content_id = kelt_client.content.create(
        content_text="This is a test article about AI and machine learning.",
        source="test",
        external_id="test_001",
        title="Test Article",
        extra={"category": "tech"},
    )
    return content_id


@pytest.fixture
def sample_feedback(kelt_client, clean_tables):
    """Create sample feedback for testing."""
    # Create content first, then record feedback on it
    content_id = kelt_client.content.create(
        content_text="Sample content for feedback",
        source="test",
    )
    feedback_id = kelt_client.atomic.feedback.record(
        signal="positive",
        content_id=content_id,
        strength=0.9,
        tags=["interesting"],
    )
    return feedback_id


# Default embedding dimensions (matches default EmbeddingConfig)
TEST_EMBEDDING_DIMS = 384


def make_test_embedding(seed: float = 0.1, dims: int = TEST_EMBEDDING_DIMS) -> list[float]:
    """Create a test embedding vector with proper dimensions.

    Args:
        seed: Base value for the embedding (varies the values).
        dims: Number of dimensions (default matches EmbeddingConfig default).

    Returns:
        List of floats with the specified dimensions.
    """
    return [seed + i * 0.001 for i in range(dims)]


@pytest.fixture
def test_embedding():
    """Factory fixture for creating test embeddings."""
    return make_test_embedding
