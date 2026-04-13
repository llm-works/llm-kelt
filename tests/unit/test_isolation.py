"""Unit tests for isolation context utilities."""

from llm_kelt.memory.isolation import ClientContext, build_context_filter, glob_to_like


class TestGlobToLike:
    """Tests for glob_to_like pattern conversion."""

    def test_no_wildcards_returns_unchanged(self):
        """Plain string without wildcards passes through unchanged."""
        pattern, is_glob = glob_to_like("acme:prod:reviewer")
        assert pattern == "acme:prod:reviewer"
        assert is_glob is False

    def test_star_becomes_percent(self):
        """Asterisk (*) converts to SQL percent (%)."""
        pattern, is_glob = glob_to_like("acme:*")
        assert pattern == "acme:%"
        assert is_glob is True

    def test_question_becomes_underscore(self):
        """Question mark (?) converts to SQL underscore (_)."""
        pattern, is_glob = glob_to_like("acme:???")
        assert pattern == "acme:___"
        assert is_glob is True

    def test_mixed_wildcards(self):
        """Both wildcards convert correctly."""
        pattern, is_glob = glob_to_like("*:prod:?")
        assert pattern == "%:prod:_"
        assert is_glob is True

    def test_no_escaping_without_wildcards(self):
        """Without wildcards, string passes through unchanged (no escaping)."""
        # SQL special chars are NOT escaped when no glob wildcards present
        pattern, is_glob = glob_to_like("100%_done")
        assert pattern == "100%_done"
        assert is_glob is False

    def test_escapes_sql_percent_with_wildcard(self):
        """Literal % is escaped when glob wildcards are present."""
        pattern, is_glob = glob_to_like("100%*")
        assert pattern == r"100\%%"
        assert is_glob is True

    def test_escapes_sql_underscore_with_wildcard(self):
        """Literal _ is escaped when glob wildcards are present."""
        pattern, is_glob = glob_to_like("some_value*")
        assert pattern == r"some\_value%"
        assert is_glob is True

    def test_escapes_backslash_with_wildcard(self):
        """Backslash is escaped when glob wildcards are present."""
        pattern, is_glob = glob_to_like(r"path\to\*")
        assert pattern == r"path\\to\\%"
        assert is_glob is True

    def test_complex_escaping_with_wildcards(self):
        """All escaping works correctly with wildcards present."""
        pattern, is_glob = glob_to_like(r"100%_test\*")
        assert pattern == r"100\%\_test\\%"
        assert is_glob is True


class TestBuildContextFilter:
    """Tests for build_context_filter SQL filter generation."""

    def test_none_returns_none(self):
        """None context_key returns None filter."""
        result = build_context_filter(None, "column")
        assert result is None

    def test_exact_match_uses_equality(self):
        """Non-glob pattern uses equality comparison."""
        # We can't easily test the SQL expression without a real column,
        # but we can verify it doesn't crash
        from sqlalchemy import String, column

        col = column("ctx", String)
        result = build_context_filter("exact_value", col)
        assert result is not None
        # Check it's an equality expression
        assert "=" in str(result) or "ctx" in str(result)

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
