"""Unit tests for LearnClient isolation context fluent API."""

from unittest.mock import MagicMock, Mock

import pytest

from llm_learn import IsolationContext
from llm_learn.client import LearnClient


class TestLearnClientIsolation:
    """Tests for LearnClient isolation context and fluent API."""

    @pytest.fixture
    def mock_database(self):
        """Create mock database."""
        db = Mock()
        db.session = MagicMock()
        db.engine = Mock()
        return db

    @pytest.fixture
    def mock_logger(self):
        """Create mock logger."""
        return Mock()

    @pytest.fixture
    def learn_client(self, mock_logger, mock_database):
        """Create LearnClient with mocked dependencies."""
        from unittest.mock import patch

        # Mock the schema verification
        with patch.object(LearnClient, "_verify_schema"):
            with patch.object(LearnClient, "_setup_stores"):
                with patch.object(LearnClient, "_setup_query_interface"):
                    context = IsolationContext(context_key="a" * 32, schema_name=None)
                    client = LearnClient(
                        database=mock_database,
                        context=context,
                        lg=mock_logger,
                    )
                    return client

    def test_context_property_returns_isolation_context(self, learn_client):
        """Test that context property returns IsolationContext."""
        context = learn_client.context

        assert isinstance(context, IsolationContext)
        assert context.context_key == "a" * 32
        assert context.schema_name is None

    def test_with_isolation_override_schema(self, learn_client, mock_logger, mock_database):
        """Test overriding just schema with with_isolation()."""
        from unittest.mock import patch

        with patch.object(LearnClient, "_verify_schema"):
            with patch.object(LearnClient, "_setup_stores"):
                with patch.object(LearnClient, "_setup_query_interface"):
                    new_client = learn_client.with_isolation(schema_name="public")

        # New client should have merged context
        assert new_client.context.context_key == "a" * 32  # Kept from original
        assert new_client.context.schema_name == "public"  # Overridden

    def test_with_isolation_override_context_key(self, learn_client, mock_logger, mock_database):
        """Test overriding just context_key with with_isolation()."""
        from unittest.mock import patch

        with patch.object(LearnClient, "_verify_schema"):
            with patch.object(LearnClient, "_setup_stores"):
                with patch.object(LearnClient, "_setup_query_interface"):
                    new_client = learn_client.with_isolation(context_key="b" * 32)

        # New client should have new context_key
        assert new_client.context.context_key == "b" * 32
        assert new_client.context.schema_name is None  # Kept from original

    def test_with_isolation_override_both(self, learn_client, mock_logger, mock_database):
        """Test overriding both fields with with_isolation()."""
        from unittest.mock import patch

        with patch.object(LearnClient, "_verify_schema"):
            with patch.object(LearnClient, "_setup_stores"):
                with patch.object(LearnClient, "_setup_query_interface"):
                    new_client = learn_client.with_isolation(
                        context_key="c" * 32, schema_name="customer_acme"
                    )

        # New client should have both overridden
        assert new_client.context.context_key == "c" * 32
        assert new_client.context.schema_name == "customer_acme"

    def test_with_isolation_preserves_other_params(self, learn_client):
        """Test that with_isolation() preserves other client parameters."""
        from unittest.mock import patch

        with patch.object(LearnClient, "_verify_schema"):
            with patch.object(LearnClient, "_setup_stores"):
                with patch.object(LearnClient, "_setup_query_interface"):
                    new_client = learn_client.with_isolation(schema_name="public")

        # Should preserve database reference
        assert new_client.database is learn_client.database

    def test_with_isolation_no_args_no_change(self, learn_client):
        """Test that with_isolation() with no args doesn't change context."""
        from unittest.mock import patch

        with patch.object(LearnClient, "_verify_schema"):
            with patch.object(LearnClient, "_setup_stores"):
                with patch.object(LearnClient, "_setup_query_interface"):
                    new_client = learn_client.with_isolation()

        # Should keep original context_key
        assert new_client.context.context_key == learn_client.context.context_key
        assert new_client.context.schema_name == learn_client.context.schema_name

    def test_with_isolation_can_clear_to_none(self, mock_logger, mock_database):
        """Test that with_isolation() can explicitly clear a field to None."""
        from unittest.mock import patch

        # Create client with both fields set
        with patch.object(LearnClient, "_verify_schema"):
            with patch.object(LearnClient, "_setup_stores"):
                with patch.object(LearnClient, "_setup_query_interface"):
                    context = IsolationContext(context_key="a" * 32, schema_name="customer_acme")
                    client = LearnClient(
                        database=mock_database,
                        context=context,
                        lg=mock_logger,
                    )

        # Clear context_key to None, keep schema_name
        with patch.object(LearnClient, "_verify_schema"):
            with patch.object(LearnClient, "_setup_stores"):
                with patch.object(LearnClient, "_setup_query_interface"):
                    new_client = client.with_isolation(context_key=None)

        # context_key should be None, schema_name preserved
        assert new_client.context.context_key is None
        assert new_client.context.schema_name == "customer_acme"


class TestIsolationContextIntegrationWithLearnClient:
    """Integration tests showing IsolationContext patterns with LearnClient."""

    def test_merge_pattern_with_dataclasses_replace(self):
        """Test the recommended pattern using dataclasses.replace()."""
        from dataclasses import replace

        # Original context
        original = IsolationContext(context_key="acme:prod", schema_name="customer_acme")

        # Override just schema (recommended pattern)
        override = replace(original, schema_name="public")

        assert override.context_key == "acme:prod"
        assert override.schema_name == "public"

    def test_partial_context_for_override(self):
        """Test creating partial context for with_isolation()."""
        # Create partial context (only schema set)
        partial = IsolationContext(schema_name="analytics")

        assert partial.context_key is None
        assert partial.schema_name == "analytics"

        # This would be merged with current context in with_isolation()
