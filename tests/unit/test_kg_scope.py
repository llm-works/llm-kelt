# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-kelt Authors

"""Unit tests for KG scope resolution helpers (no database)."""

from llm_kelt.memory.kg.store import scope_ancestors


class TestScopeAncestors:
    """scope_ancestors walks every ':' prefix, whatever the key's shape."""

    def test_global_scope(self):
        """Global scope returns only itself."""
        assert scope_ancestors("global") == ["global"]

    def test_single_level(self):
        """Single level scope includes global."""
        result = scope_ancestors("org:acme")
        assert "org:acme" in result
        assert "global" in result

    def test_two_levels(self):
        """Two level scope includes every prefix and global."""
        result = scope_ancestors("org:acme:user:alice")
        assert result == ["org:acme:user:alice", "org:acme:user", "org:acme", "org", "global"]

    def test_three_levels(self):
        """Three level scope includes all ancestors."""
        result = scope_ancestors("org:acme:team:eng:user:bob")
        assert "org:acme:team:eng:user:bob" in result
        assert "org:acme:team:eng" in result
        assert "org:acme" in result
        assert "global" in result

    def test_order_is_specific_to_general(self):
        """Ancestors are ordered from most specific to most general."""
        result = scope_ancestors("org:acme:user:alice")
        assert result.index("org:acme:user:alice") < result.index("org:acme")
        assert result.index("org:acme") < result.index("global")

    def test_odd_number_of_parts(self):
        """A key with a leading namespace part resolves to each of its prefixes."""
        assert scope_ancestors("app:org:acme:team:eng") == [
            "app:org:acme:team:eng",
            "app:org:acme:team",
            "app:org:acme",
            "app:org",
            "app",
            "global",
        ]
