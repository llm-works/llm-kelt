"""Pytest fixtures for Learn framework tests."""

import os
import socket
from pathlib import Path
from urllib.parse import urlparse

import pytest
from appinfra.config import Config
from appinfra.log import LogConfig, LoggerFactory

from llm_learn.client import LearnClient
from llm_learn.core.database import Database
from llm_learn.core.models import Base

# Import atomic memory models so they're registered with Base for migrations
from llm_learn.memory.atomic import models as atomic_models  # noqa: F401

# Import training models so they're registered with Base for migrations
from llm_learn.training import runs as training_runs  # noqa: F401

# Enable appinfra's schema isolation fixtures for parallel test execution
pytest_plugins = ["appinfra.db.pg.testing"]


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


# Find project root and config paths
PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "etc" / "llm-learn.yaml"


def _get_config_path() -> Path:
    """Get config path from env var or default."""
    config_file = os.environ.get("LEARN_TEST_CONFIG_FILE")
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
    """Auto-apply markers based on test directory."""
    for item in items:
        test_path = Path(item.fspath)
        # Determine marker from directory name
        if "unit" in test_path.parts:
            item.add_marker(pytest.mark.unit)
        elif "integration" in test_path.parts:
            item.add_marker(pytest.mark.integration)
        elif "e2e" in test_path.parts:
            item.add_marker(pytest.mark.e2e)


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
    """Load test configuration from LEARN_TEST_CONFIG_FILE or default."""
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
def pg_test_config(config):
    """Provide database config to appinfra's schema isolation fixtures.

    Checks for DATABASE_URL environment variable first (used in CI),
    otherwise falls back to config file.
    """
    database_url = os.environ.get("DATABASE_URL")
    if database_url:
        # Return dict for CI (appinfra's pg_migrate_factory expects dict-like config)
        return {
            "url": database_url,
            "create_db": True,
            "readonly": False,
            "pool_pre_ping": True,
            "extensions": ["vector"],  # pgvector for embeddings
        }

    # Fall back to config file
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


@pytest.fixture(scope="session")
def test_context(database):
    """Return a test context key for all tests."""
    # Simple hash-based context key for testing
    from hashlib import md5

    return md5(b"test:default").hexdigest()


@pytest.fixture
def learn_client(logger, database, test_context):
    """Create LearnClient for testing, scoped to test context."""
    from llm_learn import IsolationContext

    context = IsolationContext(context_key=test_context, schema_name=None)
    return LearnClient(database=database, context=context, lg=logger)


@pytest.fixture
def clean_tables(database, test_context):
    """Clean all tables before each test."""
    with database.session() as session:
        # Delete in reverse order to respect foreign keys
        for table in reversed(Base.metadata.sorted_tables):
            session.execute(table.delete())
    yield


@pytest.fixture
def sample_content(learn_client, clean_tables):
    """Create sample content for testing."""
    content_id = learn_client.content.create(
        content_text="This is a test article about AI and machine learning.",
        source="test",
        external_id="test_001",
        title="Test Article",
        metadata={"category": "tech"},
    )
    return content_id


@pytest.fixture
def sample_feedback(learn_client, clean_tables):
    """Create sample feedback for testing."""
    # Create content first, then record feedback on it
    content_id = learn_client.content.create(
        content_text="Sample content for feedback",
        source="test",
    )
    feedback_id = learn_client.feedback.record(
        signal="positive",
        content_id=content_id,
        strength=0.9,
        tags=["interesting"],
    )
    return feedback_id
