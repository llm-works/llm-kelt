"""Pytest fixtures for Learn framework tests."""

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from appinfra.config import Config
from appinfra.db.pg import PG
from appinfra.log import LogConfig, LoggerFactory

from llm_learn.client import LearnClient
from llm_learn.core.database import Database
from llm_learn.core.models import Base, Profile, Workspace


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


# Find project root
PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_PATH = PROJECT_ROOT / "etc" / "infra.yaml"


@pytest.fixture(scope="session")
def config():
    """Load test configuration."""
    return Config(str(CONFIG_PATH))


@pytest.fixture(scope="session")
def logger():
    """Create test logger."""
    log_config = LogConfig.from_params(level="debug")
    return LoggerFactory.create_root(log_config)


@pytest.fixture(scope="session")
def db_config(config):
    """Get test database configuration.

    Checks for DATABASE_URL environment variable first (used in CI),
    otherwise falls back to config file.
    """
    database_url = os.environ.get("DATABASE_URL")
    if database_url:
        # Wrap in SimpleNamespace for CI (appinfra uses attribute access)
        return SimpleNamespace(
            url=database_url,
            create_db=True,
            readonly=False,
            pool_pre_ping=True,
        )

    # Fall back to config file
    db_cfg = config.dbs.get("unittest")
    if db_cfg is None:
        pytest.skip("Database config 'dbs.unittest' not found in etc/infra.yaml")
    return db_cfg


@pytest.fixture(scope="session")
def pg(logger, db_config):
    """Create PG instance for testing."""
    return PG(logger, db_config)


@pytest.fixture(scope="session")
def database(pg):
    """Create Database wrapper and run migrations."""
    db = Database.from_pg(pg)
    db.migrate()  # Ensure tables exist
    return db


@pytest.fixture(scope="session")
def test_profile(database):
    """Create a test workspace and profile for all tests."""
    with database.session() as session:
        # Check if test workspace exists
        workspace = session.query(Workspace).filter_by(slug="test").first()
        if not workspace:
            workspace = Workspace(
                slug="test",
                name="Test Workspace",
                description="Workspace for unit tests",
            )
            session.add(workspace)
            session.flush()

        # Check if test profile exists
        profile = (
            session.query(Profile).filter_by(workspace_id=workspace.id, slug="default").first()
        )
        if not profile:
            profile = Profile(
                workspace_id=workspace.id,
                slug="default",
                name="Default Test Profile",
                description="Profile for unit tests",
            )
            session.add(profile)
            session.flush()

        return profile.id


@pytest.fixture
def learn_client(database, test_profile):
    """Create LearnClient for testing, scoped to test profile."""
    return LearnClient(profile_id=test_profile, database=database)


@pytest.fixture
def clean_tables(database, test_profile):
    """Clean all tables before each test, preserving workspace/profile."""
    with database.session() as session:
        # Delete in reverse order to respect foreign keys
        # Skip workspaces and profiles tables since we need them
        for table in reversed(Base.metadata.sorted_tables):
            if table.name not in ("workspaces", "profiles"):
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
    feedback_id = learn_client.feedback.record(
        content_text="Sample content for feedback",
        signal="positive",
        strength=0.9,
        tags=["interesting"],
    )
    return feedback_id
