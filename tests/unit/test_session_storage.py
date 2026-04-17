"""Unit tests for FileSessionStorage."""

from __future__ import annotations

from pathlib import Path

import pytest
from appinfra.log import LogConfig, LoggerFactory

from llm_kelt.conversation import Config, Conversation, Role
from llm_kelt.conversation.storage import FileSessionStorage, SessionSummary, StoredSession
from llm_kelt.core.errors import NotFoundError


@pytest.fixture
def lg():
    """Create a logger for testing."""
    log_config = LogConfig.from_params(level="warning")
    return LoggerFactory.create_root(log_config)


@pytest.fixture
def storage(tmp_path: Path, lg):
    """Create FileSessionStorage with temp directory."""
    return FileSessionStorage(lg, tmp_path)


def _make_conversation(lg, messages: list[tuple[str, str]] | None = None) -> Conversation:
    """Helper to create a conversation with optional messages."""
    conv = Conversation(lg)
    if messages:
        for role, content in messages:
            conv.add(content, role)
    return conv


# =============================================================================
# Save / Load
# =============================================================================


class TestSaveLoad:
    """Tests for save and load operations."""

    def test_save_creates_file(self, storage: FileSessionStorage, tmp_path: Path, lg):
        conv = _make_conversation(lg, [("user", "hello"), ("assistant", "hi")])
        storage.save("s1", conv)

        assert (tmp_path / "s1.json").exists()

    def test_load_roundtrip(self, storage: FileSessionStorage, lg):
        conv = _make_conversation(lg, [("user", "hello"), ("assistant", "hi")])
        storage.save("s1", conv, metadata={"model": "qwen2.5"})

        loaded = storage.load("s1")
        assert loaded.session_id == "s1"
        assert len(loaded.messages) == 2
        assert loaded.messages[0]["role"] == "user"
        assert loaded.messages[0]["content"] == "hello"
        assert loaded.metadata == {"model": "qwen2.5"}
        assert loaded.token_count == conv.token_count

    def test_load_preserves_config(self, storage: FileSessionStorage, lg):
        config = Config(max_tokens=8000, compact_threshold=0.5)
        conv = Conversation(lg, config=config)
        conv.add("test")
        storage.save("s1", conv)

        loaded = storage.load("s1")
        assert loaded.config["max_tokens"] == 8000
        assert loaded.config["compact_threshold"] == 0.5

    def test_load_not_found(self, storage: FileSessionStorage):
        with pytest.raises(NotFoundError):
            storage.load("nonexistent")

    def test_save_overwrites_preserves_created_at(self, storage: FileSessionStorage, lg):
        conv = _make_conversation(lg, [("user", "v1")])
        storage.save("s1", conv)

        loaded1 = storage.load("s1")
        created_at = loaded1.created_at

        conv2 = _make_conversation(lg, [("user", "v2")])
        storage.save("s1", conv2)

        loaded2 = storage.load("s1")
        assert loaded2.created_at == created_at
        assert loaded2.messages[0]["content"] == "v2"

    def test_save_with_tool_calls(self, storage: FileSessionStorage, lg):
        conv = Conversation(lg)
        from llm_kelt.conversation import ToolCall

        conv.add("", Role.ASSISTANT, tool_calls=[ToolCall(id="tc_1", name="search", arguments={})])
        conv.add("results", Role.TOOL, tool_call_id="tc_1")
        storage.save("s1", conv)

        loaded = storage.load("s1")
        assert loaded.messages[0]["tool_calls"] == [
            {"id": "tc_1", "name": "search", "arguments": {}}
        ]
        assert loaded.messages[1]["tool_call_id"] == "tc_1"

    def test_save_creates_directory(self, lg, tmp_path: Path):
        nested = tmp_path / "deep" / "nested" / "sessions"
        storage = FileSessionStorage(lg, nested)

        conv = _make_conversation(lg, [("user", "hello")])
        storage.save("s1", conv)

        assert nested.exists()
        assert (nested / "s1.json").exists()


# =============================================================================
# List
# =============================================================================


class TestList:
    """Tests for listing sessions."""

    def test_list_empty(self, storage: FileSessionStorage):
        assert storage.list() == []

    def test_list_returns_summaries(self, storage: FileSessionStorage, lg):
        conv1 = _make_conversation(lg, [("user", "first question")])
        conv2 = _make_conversation(lg, [("user", "second question"), ("assistant", "answer")])
        storage.save("s1", conv1)
        storage.save("s2", conv2)

        summaries = storage.list()
        assert len(summaries) == 2

        ids = {s.session_id for s in summaries}
        assert ids == {"s1", "s2"}

    def test_list_sorted_by_updated(self, storage: FileSessionStorage, lg):
        conv1 = _make_conversation(lg, [("user", "first")])
        storage.save("s1", conv1)

        conv2 = _make_conversation(lg, [("user", "second")])
        storage.save("s2", conv2)

        summaries = storage.list()
        # s2 was saved last, should be first
        assert summaries[0].session_id == "s2"

    def test_list_limit(self, storage: FileSessionStorage, lg):
        for i in range(5):
            conv = _make_conversation(lg, [("user", f"msg{i}")])
            storage.save(f"s{i}", conv)

        summaries = storage.list(limit=2)
        assert len(summaries) == 2

    def test_list_preview(self, storage: FileSessionStorage, lg):
        conv = _make_conversation(lg, [("user", "What is the weather?")])
        storage.save("s1", conv)

        summaries = storage.list()
        assert summaries[0].preview == "What is the weather?"

    def test_list_preview_truncation(self, storage: FileSessionStorage, lg):
        long_msg = "x" * 200
        conv = _make_conversation(lg, [("user", long_msg)])
        storage.save("s1", conv)

        summaries = storage.list()
        assert summaries[0].preview.endswith("...")
        assert len(summaries[0].preview) < 200

    def test_list_nonexistent_directory(self, lg):
        storage = FileSessionStorage(lg, "/tmp/nonexistent_dir_xyz")
        assert storage.list() == []

    def test_list_skips_corrupt_files(self, storage: FileSessionStorage, tmp_path: Path, lg):
        # Write a valid session
        conv = _make_conversation(lg, [("user", "good")])
        storage.save("good", conv)

        # Write a corrupt file
        (tmp_path / "bad.json").write_text("not json{{{")

        summaries = storage.list()
        assert len(summaries) == 1
        assert summaries[0].session_id == "good"


# =============================================================================
# Delete
# =============================================================================


class TestDelete:
    """Tests for deleting sessions."""

    def test_delete_existing(self, storage: FileSessionStorage, tmp_path: Path, lg):
        conv = _make_conversation(lg, [("user", "hello")])
        storage.save("s1", conv)

        assert storage.delete("s1") is True
        assert not (tmp_path / "s1.json").exists()

    def test_delete_nonexistent(self, storage: FileSessionStorage):
        assert storage.delete("nonexistent") is False

    def test_delete_then_load_raises(self, storage: FileSessionStorage, lg):
        conv = _make_conversation(lg, [("user", "hello")])
        storage.save("s1", conv)
        storage.delete("s1")

        with pytest.raises(NotFoundError):
            storage.load("s1")


# =============================================================================
# StoredSession / SessionSummary
# =============================================================================


class TestDataModels:
    """Tests for StoredSession and SessionSummary FieldDicts."""

    def test_stored_session_defaults(self):
        s = StoredSession(session_id="s1")
        assert s.session_id == "s1"
        assert s.messages == []
        assert s.metadata == {}
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
