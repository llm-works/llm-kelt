# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-kelt Authors

"""Unit tests for KG scope resolution helpers (no database)."""

from llm_kelt.memory.kg.store import scope_ancestors


class TestScopeAncestors:
    """scope_ancestors walks every ':' prefix, whatever the key's shape."""

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
