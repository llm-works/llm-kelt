"""Unit tests for ScopedClient and Client.with_schema()."""

from unittest.mock import MagicMock, Mock, patch

import pytest

from llm_kelt import ClientContext
from llm_kelt.client import Client
from llm_kelt.scoped_client import ScopedClient


class TestClientWithSchema:
    """Tests for Client.with_schema() method."""

    @pytest.fixture
    def mock_database(self):
        """Create mock database."""
        db = Mock()
        db.session = MagicMock()
        db.engine = Mock()
        db.schema = None
        return db

    @pytest.fixture
    def mock_logger(self):
        """Create mock logger."""
        return Mock()

    @pytest.fixture
    def kelt_client(self, mock_logger, mock_database):
        """Create Client with mocked dependencies."""
        with patch.object(Client, "_verify_schema"):
            with patch.object(Client, "_setup_stores"):
                with patch.object(Client, "_setup_query_interface"):
                    context = ClientContext(context_key="test-agent")
                    client = Client(
                        database=mock_database,
                        context=context,
                        lg=mock_logger,
                        ensure_schema=True,
                    )
                    return client

    def test_with_schema_returns_scoped_client(self, kelt_client):
        """Test that with_schema() returns a ScopedClient."""
        scoped = kelt_client.with_schema("my_schema")

        assert isinstance(scoped, ScopedClient)
        assert scoped.schema_name == "my_schema"

    def test_with_schema_multiple_calls_independent(self, kelt_client):
        """Test that multiple with_schema() calls return independent clients."""
        scoped_a = kelt_client.with_schema("schema_a")
        scoped_b = kelt_client.with_schema("schema_b")

        assert scoped_a.schema_name == "schema_a"
        assert scoped_b.schema_name == "schema_b"
        assert scoped_a is not scoped_b

    def test_with_schema_inherits_ensure_schema(self, kelt_client):
        """Test that ScopedClient inherits ensure_schema setting from parent."""
        scoped = kelt_client.with_schema("my_schema")

        # Access private attribute to verify
        assert scoped._ensure_schema is True

    def test_with_schema_inherits_context_key(self, kelt_client):
        """Test that ScopedClient uses parent's context_key."""
        scoped = kelt_client.with_schema("my_schema")

        # The context_key is used when Protocol is created in _ensure_initialized
        assert scoped._parent._context.context_key == "test-agent"


class TestScopedClientLazyInit:
    """Tests for ScopedClient lazy initialization."""

    @pytest.fixture
    def mock_logger(self):
        """Create mock logger."""
        return Mock()

    @pytest.fixture
    def mock_parent(self, mock_logger):
        """Create mock parent Client."""
        parent = Mock()
        parent._lg = mock_logger
        parent._context = ClientContext(context_key="test-agent")
        parent._embedder = None

        # Mock database and scoped database
        scoped_db = Mock()
        scoped_db.session = MagicMock()
        scoped_db.engine = Mock()
        scoped_db.schema = "my_schema"
        parent._db = Mock()
        parent._db.scoped.return_value = scoped_db

        return parent

    def test_not_initialized_on_creation(self, mock_logger, mock_parent):
        """Test that ScopedClient is not initialized on creation."""
        scoped = ScopedClient(
            lg=mock_logger,
            parent=mock_parent,
            schema_name="my_schema",
            ensure_schema=True,
        )

        assert scoped._initialized is False
        assert scoped._scoped_db is None
        assert scoped._atomic is None
        mock_parent._db.scoped.assert_not_called()

    def test_initialized_on_atomic_access(self, mock_logger, mock_parent):
        """Test that accessing atomic triggers initialization."""
        scoped = ScopedClient(
            lg=mock_logger,
            parent=mock_parent,
            schema_name="my_schema",
            ensure_schema=True,
        )

        # Patch SchemaManager to avoid actual migrations
        with patch("llm_kelt.scoped_client.SchemaManager"):
            _ = scoped.atomic

        assert scoped._initialized is True
        mock_parent._db.scoped.assert_called_once_with("my_schema")

    def test_ensure_schema_called_when_enabled(self, mock_logger, mock_parent):
        """Test that ensure_schema creates schema when enabled."""
        scoped = ScopedClient(
            lg=mock_logger,
            parent=mock_parent,
            schema_name="my_schema",
            ensure_schema=True,
        )

        with patch("llm_kelt.scoped_client.SchemaManager") as mock_schema_manager_cls:
            mock_manager = Mock()
            mock_schema_manager_cls.return_value = mock_manager

            _ = scoped.atomic

        # Verify schema creation was called
        scoped_db = mock_parent._db.scoped.return_value
        scoped_db.ensure_schema.assert_called_once()
        mock_manager.ensure_schema.assert_called_once()

    def test_ensure_schema_not_called_when_disabled(self, mock_logger, mock_parent):
        """Test that ensure_schema skips creation when disabled."""
        scoped = ScopedClient(
            lg=mock_logger,
            parent=mock_parent,
            schema_name="my_schema",
            ensure_schema=False,
        )

        with patch("llm_kelt.scoped_client.SchemaManager") as mock_schema_manager_cls:
            _ = scoped.atomic

        # Verify schema creation was NOT called
        scoped_db = mock_parent._db.scoped.return_value
        scoped_db.ensure_schema.assert_not_called()
        mock_schema_manager_cls.assert_not_called()

    def test_initialization_only_happens_once(self, mock_logger, mock_parent):
        """Test that repeated atomic access doesn't re-initialize."""
        scoped = ScopedClient(
            lg=mock_logger,
            parent=mock_parent,
            schema_name="my_schema",
            ensure_schema=True,
        )

        with patch("llm_kelt.scoped_client.SchemaManager"):
            _ = scoped.atomic
            _ = scoped.atomic
            _ = scoped.atomic

        # Should only be called once
        mock_parent._db.scoped.assert_called_once()


class TestScopedClientSchemaName:
    """Tests for ScopedClient.schema_name property."""

    def test_schema_name_available_before_init(self):
        """Test that schema_name is available without triggering initialization."""
        mock_parent = Mock()
        scoped = ScopedClient(
            lg=Mock(),
            parent=mock_parent,
            schema_name="early_access_schema",
            ensure_schema=True,
        )

        # schema_name should work without initialization
        assert scoped.schema_name == "early_access_schema"
        assert scoped._initialized is False
