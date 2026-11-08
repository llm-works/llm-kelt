# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-kelt Authors

"""Unit tests for key and pattern translation (no database)."""

import pytest
from sqlalchemy import String, column

from llm_kelt.core.errors import ValidationError
from llm_kelt.core.keys import (
    KeyMatch,
    is_key_pattern,
    key_filter,
    key_sql,
    parse_key,
    require_exact_key,
)


def _sql(expr) -> tuple[str, dict]:
    """SQL text with named placeholders, and the values bound to them."""
    compiled = expr.compile()
    return str(compiled), compiled.params


class TestParseKey:
    """parse_key reduces each key form to SQL terms."""

    def test_exact_key(self):
        assert parse_key("acme:prod") == KeyMatch(equals="acme:prod")

    def test_exact_key_keeps_like_wildcards_literal(self):
        """% and _ in a key are ordinary characters, not wildcards."""
        assert parse_key("100%_done") == KeyMatch(equals="100%_done")

    def test_star(self):
        assert parse_key("acme:*") == KeyMatch(likes=("acme:%",))

    def test_question_mark(self):
        assert parse_key("acme:???") == KeyMatch(likes=("acme:___",))

    def test_glob_escapes_like_wildcards(self):
        assert parse_key("some_value*") == KeyMatch(likes=(r"some\_value%",))
        assert parse_key("100%*") == KeyMatch(likes=(r"100\%%",))

    def test_glob_escapes_backslash(self):
        assert parse_key(r"path\to\*") == KeyMatch(likes=(r"path\\to\\%",))

    def test_subtree_is_key_or_below(self):
        """K:** matches K itself and keys below K:, not keys that merely start with K."""
        assert parse_key("acme:prod:**") == KeyMatch(equals="acme:prod", likes=("acme:prod:%",))

    def test_subtree_escapes_base(self):
        assert parse_key("team_a:**") == KeyMatch(equals="team_a", likes=(r"team\_a:%",))

    def test_subtree_of_glob(self):
        assert parse_key("acme:*:eng:**") == KeyMatch(likes=("acme:%:eng", "acme:%:eng:%"))


class TestKeyPatternChecks:
    """Patterns select keys; writes refuse them."""

    @pytest.mark.parametrize("key", ["acme:*", "acme:?", "acme:**"])
    def test_patterns(self, key):
        assert is_key_pattern(key)
        with pytest.raises(ValidationError, match="cannot be written"):
            require_exact_key(key)

    @pytest.mark.parametrize("key", ["acme:prod", "100%_done", "global"])
    def test_keys(self, key):
        assert not is_key_pattern(key)
        assert require_exact_key(key) == key


class TestKeyFilter:
    """key_filter renders each form as SQL."""

    col = column("ctx", String)

    def test_exact_key_is_equality(self):
        assert _sql(key_filter("acme:prod", self.col)) == ("ctx = :ctx_1", {"ctx_1": "acme:prod"})

    def test_glob_is_like(self):
        assert _sql(key_filter("acme:*", self.col)) == (
            "ctx LIKE :ctx_1 ESCAPE '\\'",
            {"ctx_1": "acme:%"},
        )

    def test_subtree_is_equality_or_like(self):
        assert _sql(key_filter("acme:**", self.col)) == (
            "ctx = :ctx_1 OR ctx LIKE :ctx_2 ESCAPE '\\'",
            {"ctx_1": "acme", "ctx_2": "acme:%"},
        )


class TestKeySql:
    """key_sql renders the same terms as raw SQL with bind parameters."""

    def test_exact_key(self):
        assert key_sql("acme", "r.ctx", "k") == ("(r.ctx = :k)", {"k": "acme"})

    def test_subtree(self):
        assert key_sql("acme:**", "r.ctx", "k") == (
            "(r.ctx = :k OR r.ctx LIKE :k_0 ESCAPE '\\')",
            {"k": "acme", "k_0": "acme:%"},
        )
