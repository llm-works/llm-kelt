"""Unit tests for conversation types, session, and token estimation."""

from __future__ import annotations

import dataclasses
from unittest.mock import MagicMock

import pytest

from llm_kelt.conversation import (
    AsyncCompactor,
    Compactor,
    Config,
    Conversation,
    Message,
    Role,
    ToolCall,
    estimate_message_tokens,
    estimate_tokens,
)

# =============================================================================
# Types
# =============================================================================


class TestMessage:
    """Tests for Message dataclass (from saia)."""

    def test_basic_message(self):
        msg = Message(role="user", content="hello")
        assert msg.role == "user"
        assert msg.content == "hello"
        assert msg.tool_calls is None
        assert msg.tool_call_id is None

    def test_assistant_with_tool_calls(self):
        calls = [ToolCall(id="tc_1", name="search", arguments={"q": "test"})]
        msg = Message(role="assistant", content="", tool_calls=calls)
        assert msg.tool_calls == calls

    def test_tool_result(self):
        msg = Message(role="tool", content="result data", tool_call_id="tc_1")
        assert msg.tool_call_id == "tc_1"

    def test_message_as_dict(self):
        msg = Message(role="user", content="hello")
        d = dataclasses.asdict(msg)
        assert d["role"] == "user"
        assert d["content"] == "hello"

    def test_message_dict_roundtrip(self):
        calls = [ToolCall(id="1", name="search", arguments={})]
        msg = Message(role="assistant", content="hi", tool_calls=calls)
        d = dataclasses.asdict(msg)
        # Roundtrip requires reconstructing ToolCall from dict
        d["tool_calls"] = [ToolCall(**tc) for tc in d["tool_calls"]]
        restored = Message(**d)
        assert restored.role == msg.role
        assert restored.content == msg.content
        assert restored.tool_calls == msg.tool_calls


class TestToolCall:
    """Tests for ToolCall dataclass (from saia)."""

    def test_basic_tool_call(self):
        tc = ToolCall(id="tc_1", name="search", arguments={"q": "test"})
        assert tc.id == "tc_1"
        assert tc.name == "search"
        assert tc.arguments == {"q": "test"}

    def test_tool_call_as_dict(self):
        tc = ToolCall(id="tc_1", name="search", arguments={})
        d = dataclasses.asdict(tc)
        assert d["id"] == "tc_1"


class TestRole:
    """Tests for Role StrEnum."""

    def test_role_values(self):
        assert Role.SYSTEM == "system"
        assert Role.USER == "user"
        assert Role.ASSISTANT == "assistant"
        assert Role.TOOL == "tool"

    def test_role_string_comparison(self):
        """StrEnum should compare equal to plain strings."""
        assert Role.USER == "user"
        assert "assistant" == Role.ASSISTANT


# =============================================================================
# Token Estimation
# =============================================================================


class TestTokenEstimation:
    """Tests for token estimation utilities."""

    def test_empty_string(self):
        assert estimate_tokens("") == 0

    def test_basic_estimation(self):
        # 20 chars / 4 chars_per_token = 5
        assert estimate_tokens("a" * 20) == 5

    def test_minimum_one_token(self):
        assert estimate_tokens("hi") == 1

    def test_message_tokens_include_overhead(self):
        # 20 chars = 5 tokens + 4 overhead = 9
        assert estimate_message_tokens("user", "a" * 20) == 9

    def test_message_tokens_empty_content(self):
        # 0 content tokens + 4 overhead = 4
        assert estimate_message_tokens("user", "") == 4

    def test_message_tokens_with_tool_calls(self):
        calls = [ToolCall(id="tc_1", name="search", arguments={"q": "test"})]
        tokens = estimate_message_tokens("assistant", "", calls)
        assert tokens > estimate_message_tokens("assistant", "")


# =============================================================================
# Conversation
# =============================================================================


class TestConversation:
    """Tests for Conversation session management."""

    def test_empty_conversation(self, lg):
        conv = Conversation(lg)
        assert conv.message_count == 0
        assert conv.token_count == 0
        assert len(conv) == 0
        assert conv.messages == []

    def test_add_messages(self, lg):
        conv = Conversation(lg)
        conv.add("You are helpful.", Role.SYSTEM)
        conv.add("Hello")
        conv.add("Hi there", Role.ASSISTANT)

        assert conv.message_count == 3
        assert conv.messages[0].role == "system"
        assert conv.messages[1].role == "user"
        assert conv.messages[2].role == "assistant"

    def test_add_tool_messages(self, lg):
        conv = Conversation(lg)
        conv.add("", Role.ASSISTANT, tool_calls=[ToolCall(id="tc_1", name="search", arguments={})])
        conv.add("results", Role.TOOL, tool_call_id="tc_1")

        assert conv.message_count == 2
        assert conv.messages[0].tool_calls is not None
        assert conv.messages[1].tool_call_id == "tc_1"

    def test_default_role_is_user(self, lg):
        conv = Conversation(lg)
        conv.add("hello")
        assert conv.messages[0].role == "user"

    def test_token_tracking(self, lg):
        conv = Conversation(lg)
        conv.add("hello")
        initial = conv.token_count
        assert initial > 0

        conv.add("world", Role.ASSISTANT)
        assert conv.token_count > initial

    def test_usage_ratio(self, lg):
        config = Config(max_tokens=100)
        conv = Conversation(lg, config=config)
        assert conv.usage_ratio == 0.0

        conv.add("test")
        assert 0.0 < conv.usage_ratio < 1.0

    def test_usage_ratio_zero_max(self, lg):
        config = Config(max_tokens=0)
        conv = Conversation(lg, config=config)
        assert conv.usage_ratio == 0.0

    def test_needs_compaction(self, lg):
        config = Config(max_tokens=50, compact_threshold=0.5)
        conv = Conversation(lg, config=config)

        # Add enough content to exceed threshold
        conv.add("a" * 200)
        assert conv.needs_compaction()

    def test_no_compaction_needed(self, lg):
        config = Config(max_tokens=100000)
        conv = Conversation(lg, config=config)
        conv.add("short message")
        assert not conv.needs_compaction()

    def test_clear(self, lg):
        conv = Conversation(lg)
        conv.add("hello")
        conv.add("hi", Role.ASSISTANT)
        conv.clear()

        assert conv.message_count == 0
        assert conv.token_count == 0

    def test_replace_messages(self, lg):
        conv = Conversation(lg)
        conv.add("old message")
        conv.add("old response", Role.ASSISTANT)

        new_msgs = [Message(role="user", content="new")]
        conv.replace_messages(new_msgs)

        assert conv.message_count == 1
        assert conv.messages[0].content == "new"
        # Token count should be recalculated
        assert conv.token_count > 0

    def test_get_system_message(self, lg):
        conv = Conversation(lg)
        conv.add("system prompt", Role.SYSTEM)
        conv.add("hello")

        sys_msg = conv.get_system_message()
        assert sys_msg is not None
        assert sys_msg.content == "system prompt"

    def test_get_system_message_none(self, lg):
        conv = Conversation(lg)
        conv.add("hello")
        assert conv.get_system_message() is None

    def test_messages_returns_copy(self, lg):
        conv = Conversation(lg)
        conv.add("hello")
        msgs = conv.messages
        msgs.append(Message(role="user", content="extra"))
        assert conv.message_count == 1  # Original unchanged

    def test_as_messages_returns_view(self, lg):
        """as_messages() returns the live internal list (ConversationLike contract)."""
        conv = Conversation(lg)
        conv.add("hello")
        view = conv.as_messages()
        assert view is conv._messages

    def test_append_protocol(self, lg):
        """append() implements ConversationLike protocol."""
        conv = Conversation(lg)
        conv.append(Message(role="user", content="hello"))
        assert conv.message_count == 1
        assert conv.token_count > 0

    def test_auto_compaction_with_injected_compactor(self, lg):
        """Test that compaction fires automatically when compactor is set."""
        from llm_kelt.conversation import SlidingWindowCompactor

        config = Config(max_tokens=50, compact_threshold=0.5, min_recent_messages=1)
        conv = Conversation(lg, config=config, compactor=SlidingWindowCompactor())

        conv.add("first message")
        conv.add("second message", Role.ASSISTANT)
        conv.add("a" * 200)  # Should trigger compaction

        # Should have been compacted — only recent messages preserved
        assert conv.message_count <= 2

    def test_no_auto_compaction_without_compactor(self, lg):
        config = Config(max_tokens=50, compact_threshold=0.5)
        conv = Conversation(lg, config=config)

        conv.add("a" * 200)
        # needs_compaction is True, but no compactor means no auto-compact
        assert conv.needs_compaction()
        assert conv.message_count == 1

    def test_messages_as_dicts_strips_nones(self, lg):
        conv = Conversation(lg)
        conv.add("hello")
        dicts = conv.messages_as_dicts()
        assert "tool_calls" not in dicts[0]
        assert "tool_call_id" not in dicts[0]
        assert dicts[0]["role"] == "user"
        assert dicts[0]["content"] == "hello"


class TestSplitForCompaction:
    """Tests for Conversation.split_for_compaction()."""

    def test_split_basic(self, lg):
        config = Config(min_recent_messages=2)
        conv = Conversation(lg, config=config)
        conv.add("msg1")
        conv.add("resp1", Role.ASSISTANT)
        conv.add("msg2")
        conv.add("resp2", Role.ASSISTANT)

        to_compact, preserved = conv.split_for_compaction()
        assert len(to_compact) == 2
        assert len(preserved) == 2

    def test_split_preserves_system(self, lg):
        config = Config(min_recent_messages=2, preserve_system=True)
        conv = Conversation(lg, config=config)
        conv.add("system", Role.SYSTEM)
        conv.add("msg1")
        conv.add("resp1", Role.ASSISTANT)
        conv.add("msg2")
        conv.add("resp2", Role.ASSISTANT)

        to_compact, preserved = conv.split_for_compaction()
        assert len(to_compact) == 2
        # preserved = system + 2 recent
        assert len(preserved) == 3
        assert preserved[0].role == "system"

    def test_split_not_enough_messages(self, lg):
        config = Config(min_recent_messages=4)
        conv = Conversation(lg, config=config)
        conv.add("msg1")
        conv.add("resp1", Role.ASSISTANT)

        to_compact, preserved = conv.split_for_compaction()
        assert len(to_compact) == 0
        assert len(preserved) == 2

    def test_split_no_preserve_system(self, lg):
        config = Config(min_recent_messages=2, preserve_system=False)
        conv = Conversation(lg, config=config)
        conv.add("system", Role.SYSTEM)
        conv.add("msg1")
        conv.add("resp1", Role.ASSISTANT)
        conv.add("msg2")
        conv.add("resp2", Role.ASSISTANT)

        to_compact, preserved = conv.split_for_compaction()
        # System message is included in pool when not preserved
        # separately, so it counts as a regular message
        assert len(to_compact) == 3  # system + msg1 + resp1
        assert len(preserved) == 2


class TestConfig:
    """Tests for Config FieldDict."""

    def test_defaults(self):
        config = Config()
        assert config.max_tokens == 32000
        assert config.compact_threshold == 0.8
        assert config.preserve_system is True
        assert config.min_recent_messages == 4

    def test_custom_values(self):
        config = Config(max_tokens=8000, compact_threshold=0.5)
        assert config.max_tokens == 8000
        assert config.compact_threshold == 0.5

    def test_config_is_dict(self):
        config = Config(max_tokens=8000)
        d = dict(config)
        assert d["max_tokens"] == 8000


class TestAsyncCompaction:
    """Tests for async compaction support."""

    def test_async_compactor_warning_on_construction(self):
        class MockAsyncCompactor(AsyncCompactor):
            async def compact(self, conversation: Conversation) -> None:
                pass

        mock_lg = MagicMock()
        compactor = MockAsyncCompactor()
        Conversation(mock_lg, compactor=compactor)

        mock_lg.warning.assert_called_once()
        call_args = mock_lg.warning.call_args
        assert "AsyncCompactor detected" in call_args[0][0]
        assert "append_async" in call_args[0][0]

    def test_sync_append_raises_with_async_compactor(self, lg):
        class MockAsyncCompactor(AsyncCompactor):
            async def compact(self, conversation: Conversation) -> None:
                pass

        config = Config(max_tokens=50, compact_threshold=0.5)
        compactor = MockAsyncCompactor()
        conv = Conversation(lg, config=config, compactor=compactor)

        conv.append(Message(role="user", content="short"))

        with pytest.raises(RuntimeError, match="append_async"):
            conv.append(Message(role="user", content="a" * 200))

        # Verify state was not mutated by the failed append
        assert conv.message_count == 1
        assert conv.messages[0].content == "short"

    async def test_append_async_with_sync_compactor(self, lg):
        class MockSyncCompactor(Compactor):
            def __init__(self):
                self.called = False

            def compact(self, conversation: Conversation) -> None:
                self.called = True
                msgs = conversation.messages[-2:]
                conversation.replace_messages(msgs)

        config = Config(max_tokens=50, compact_threshold=0.5)
        compactor = MockSyncCompactor()
        conv = Conversation(lg, config=config, compactor=compactor)

        await conv.append_async(Message(role="user", content="short"))
        await conv.append_async(Message(role="user", content="a" * 200))

        assert compactor.called
        assert conv.message_count == 2

    async def test_append_async_with_async_compactor(self, lg):
        class MockAsyncCompactor(AsyncCompactor):
            def __init__(self):
                self.called = False

            async def compact(self, conversation: Conversation) -> None:
                self.called = True
                msgs = conversation.messages[-2:]
                conversation.replace_messages(msgs)

        config = Config(max_tokens=50, compact_threshold=0.5)
        compactor = MockAsyncCompactor()
        conv = Conversation(lg, config=config, compactor=compactor)

        await conv.append_async(Message(role="user", content="short"))
        await conv.append_async(Message(role="user", content="a" * 200))

        assert compactor.called
        assert conv.message_count == 2

    def test_sync_compactor_no_warning(self):
        class MockSyncCompactor(Compactor):
            def compact(self, conversation: Conversation) -> None:
                pass

        mock_lg = MagicMock()
        compactor = MockSyncCompactor()
        Conversation(mock_lg, compactor=compactor)

        mock_lg.warning.assert_not_called()
