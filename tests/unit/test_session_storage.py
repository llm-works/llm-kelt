# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-kelt Authors

"""Unit tests for FileSessionStorage."""

from __future__ import annotations

from llm_kelt.conversation.storage import SessionSummary, StoredSession

# =============================================================================
# Save / Load
# =============================================================================


# =============================================================================
# List
# =============================================================================


# =============================================================================
# Delete
# =============================================================================


# =============================================================================
# StoredSession / SessionSummary
# =============================================================================


class TestDataModels:
    """Tests for StoredSession and SessionSummary FieldDicts."""

    def test_stored_session_defaults(self):
        s = StoredSession(session_id="s1")
        assert s.session_id == "s1"
        assert s.messages == []
        assert s.extra == {}
        assert s.token_count == 0

    def test_stored_session_is_dict(self):
        s = StoredSession(session_id="s1", token_count=100)
        d = dict(s)
        assert d["session_id"] == "s1"
        assert d["token_count"] == 100

    def test_session_summary_defaults(self):
        s = SessionSummary(session_id="s1")
        assert s.message_count == 0
        assert s.preview == ""

    def test_session_summary_is_dict(self):
        s = SessionSummary(session_id="s1", message_count=5, preview="hello")
        d = dict(s)
        assert d["message_count"] == 5
