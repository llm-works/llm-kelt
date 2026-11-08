# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-kelt Authors

"""Unit tests for isolation context utilities."""

from llm_kelt.memory.isolation import ClientContext, build_context_filter


class TestBuildContextFilter:
    """Tests for build_context_filter SQL filter generation."""

    def test_none_returns_none(self):
        """None context_key returns None filter."""
        result = build_context_filter(None, "column")
        assert result is None

    def test_exact_match_uses_equality(self):
        """Non-glob pattern uses equality comparison, not LIKE."""
        from sqlalchemy import String, column

        col = column("ctx", String)
        result = build_context_filter("exact_value", col)
        assert result is not None
        # Verify it's equality (=) not LIKE
        result_str = str(result).upper()
        assert "LIKE" not in result_str
        assert "=" in str(result)

    def test_glob_pattern_uses_like(self):
        """Glob pattern uses LIKE comparison."""
        from sqlalchemy import String, column

        col = column("ctx", String)
        result = build_context_filter("prefix:*", col)
        assert result is not None
        assert "LIKE" in str(result).upper()


class TestClientContext:
    """Tests for ClientContext dataclass."""

    def test_default_values(self):
        """Default context has None for both fields."""
        ctx = ClientContext()
        assert ctx.context_key is None
        assert ctx.schema_name is None

    def test_with_context_key(self):
        """Can set context_key."""
        ctx = ClientContext(context_key="my:context")
        assert ctx.context_key == "my:context"
        assert ctx.schema_name is None

    def test_with_schema_name(self):
        """Can set schema_name."""
        ctx = ClientContext(schema_name="tenant_schema")
        assert ctx.context_key is None
        assert ctx.schema_name == "tenant_schema"

    def test_with_both(self):
        """Can set both fields."""
        ctx = ClientContext(context_key="my:context", schema_name="tenant_schema")
        assert ctx.context_key == "my:context"
        assert ctx.schema_name == "tenant_schema"
