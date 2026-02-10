"""Unit tests for IsolationContext."""

from llm_learn import IsolationContext


class TestIsolationContext:
    """Tests for IsolationContext dataclass."""

    def test_create_with_both_fields(self):
        """Test creating context with both fields specified."""
        context = IsolationContext(context_key="acme:prod:reviewer", schema_name="customer_acme")

        assert context.context_key == "acme:prod:reviewer"
        assert context.schema_name == "customer_acme"

    def test_create_with_context_key_only(self):
        """Test creating context with only context_key (schema defaults to None)."""
        context = IsolationContext(context_key="acme:prod")

        assert context.context_key == "acme:prod"
        assert context.schema_name is None

    def test_create_with_schema_only(self):
        """Test creating context with only schema (no filtering)."""
        context = IsolationContext(schema_name="customer_acme")

        assert context.context_key is None
        assert context.schema_name == "customer_acme"

    def test_create_empty_context(self):
        """Test creating empty context (single tenant, simplest case)."""
        context = IsolationContext()

        assert context.context_key is None
        assert context.schema_name is None

    def test_context_key_formats(self):
        """Test various context_key formats are accepted."""
        formats = [
            "simple",
            "acme:prod",
            "customer_123",
            "customer:environment:agent",
            "a" * 255,  # Long key
        ]

        for key in formats:
            context = IsolationContext(context_key=key)
            assert context.context_key == key

    def test_schema_name_formats(self):
        """Test various schema name formats are accepted."""
        schemas = [
            "public",
            "customer_acme",
            "schema_123",
            "my_schema",
        ]

        for schema in schemas:
            context = IsolationContext(schema_name=schema)
            assert context.schema_name == schema

    def test_dataclass_equality(self):
        """Test dataclass equality."""
        ctx1 = IsolationContext(context_key="acme", schema_name="public")
        ctx2 = IsolationContext(context_key="acme", schema_name="public")
        ctx3 = IsolationContext(context_key="other", schema_name="public")

        assert ctx1 == ctx2
        assert ctx1 != ctx3

    def test_dataclass_repr(self):
        """Test dataclass repr."""
        context = IsolationContext(context_key="acme", schema_name="public")
        repr_str = repr(context)

        assert "IsolationContext" in repr_str
        assert "acme" in repr_str
        assert "public" in repr_str

    def test_immutable_after_creation(self):
        """Test that context fields can be modified (dataclass not frozen)."""
        context = IsolationContext(context_key="acme", schema_name="public")

        # Should be mutable (not frozen)
        context.context_key = "updated"
        assert context.context_key == "updated"

    def test_use_cases(self):
        """Test various use cases with appropriate contexts."""
        # Use case 1: Single agent, entire DB
        single_tenant = IsolationContext()
        assert single_tenant.context_key is None
        assert single_tenant.schema_name is None

        # Use case 2: Multi-tenant, shared tables
        multi_tenant_shared = IsolationContext(context_key="customer_123", schema_name="public")
        assert multi_tenant_shared.context_key == "customer_123"
        assert multi_tenant_shared.schema_name == "public"

        # Use case 3: Multi-tenant, isolated schemas
        multi_tenant_isolated = IsolationContext(
            context_key="customer_123", schema_name="customer_123"
        )
        assert multi_tenant_isolated.context_key == "customer_123"
        assert multi_tenant_isolated.schema_name == "customer_123"

        # Use case 4: Query different schema, no filtering
        cross_schema = IsolationContext(schema_name="analytics")
        assert cross_schema.context_key is None
        assert cross_schema.schema_name == "analytics"


class TestIsolationContextIntegration:
    """Integration tests showing how IsolationContext would be used."""

    def test_filtering_logic_with_context_key(self):
        """Test how filtering logic would work with context_key."""
        context = IsolationContext(context_key="acme:prod")

        # Simulated query logic
        filters = []
        if context.context_key is not None:
            filters.append(f"context_key = '{context.context_key}'")

        assert len(filters) == 1
        assert filters[0] == "context_key = 'acme:prod'"

    def test_filtering_logic_without_context_key(self):
        """Test how filtering logic would work without context_key."""
        context = IsolationContext()

        # Simulated query logic
        filters = []
        if context.context_key is not None:
            filters.append(f"context_key = '{context.context_key}'")

        # No filters applied - returns all data
        assert len(filters) == 0

    def test_schema_resolution(self):
        """Test how schema resolution would work."""
        # With schema specified
        context_with_schema = IsolationContext(schema_name="customer_acme")
        schema = context_with_schema.schema_name or "public"
        assert schema == "customer_acme"

        # Without schema (defaults to public)
        context_without_schema = IsolationContext()
        schema = context_without_schema.schema_name or "public"
        assert schema == "public"

    def test_context_override_pattern(self):
        """Test pattern for overriding context fields."""
        from dataclasses import replace

        # Original context
        original = IsolationContext(context_key="acme:prod", schema_name="customer_acme")

        # Override just schema
        with_public = replace(original, schema_name="public")
        assert with_public.context_key == "acme:prod"
        assert with_public.schema_name == "public"

        # Override just context_key
        with_other_context = replace(original, context_key="other:key")
        assert with_other_context.context_key == "other:key"
        assert with_other_context.schema_name == "customer_acme"

        # Override both
        completely_different = replace(original, context_key="new:key", schema_name="public")
        assert completely_different.context_key == "new:key"
        assert completely_different.schema_name == "public"
