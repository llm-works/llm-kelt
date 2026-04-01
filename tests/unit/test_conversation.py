"""Unit tests for conversation types, session, and token estimation."""

from __future__ import annotations

from llm_kelt.conversation import (
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
    """Tests for Message FieldDict."""

    def test_basic_message(self):
        msg = Message(role="user", content="hello")
        assert msg.role == "user"
        assert msg.content == "hello"
        assert msg.tool_calls is None
        assert msg.tool_call_id is None

    def test_assistant_with_tool_calls(self):
        calls = [{"id": "tc_1", "name": "search", "arguments": {"q": "test"}}]
        msg = Message(role="assistant", content="", tool_calls=calls)
        assert msg.tool_calls == calls

    def test_tool_result(self):
        msg = Message(role="tool", content="result data", tool_call_id="tc_1")
        assert msg.tool_call_id == "tc_1"

    def test_message_is_dict(self):
        """Message is a FieldDict, so it should behave as a dict."""
        msg = Message(role="user", content="hello")
        assert dict(msg)["role"] == "user"
        assert dict(msg)["content"] == "hello"

    def test_message_dict_roundtrip(self):
        msg = Message(role="assistant", content="hi", tool_calls=[{"id": "1"}])
        d = dict(msg)
        restored = Message(**d)
        assert restored.role == msg.role
        assert restored.content == msg.content
        assert restored.tool_calls == msg.tool_calls


class TestToolCall:
    """Tests for ToolCall FieldDict."""

    def test_basic_tool_call(self):
        tc = ToolCall(id="tc_1", name="search", arguments={"q": "test"})
        assert tc.id == "tc_1"
        assert tc.name == "search"
        assert tc.arguments == {"q": "test"}

    def test_default_arguments(self):
        tc = ToolCall(id="tc_1", name="noop")
        assert tc.arguments == {}

    def test_tool_call_is_dict(self):
        tc = ToolCall(id="tc_1", name="search")
        assert dict(tc)["id"] == "tc_1"


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


# =============================================================================
# Conversation
# =============================================================================


class TestConversation:
    """Tests for Conversation session management."""

    def test_empty_conversation(self):
        conv = Conversation()
        assert conv.message_count == 0
        assert conv.token_count == 0
        assert len(conv) == 0
        assert conv.messages == []

    def test_add_messages(self):
        conv = Conversation()
        conv.add("You are helpful.", Role.SYSTEM)
        conv.add("Hello")
        conv.add("Hi there", Role.ASSISTANT)

        assert conv.message_count == 3
        assert conv.messages[0].role == "system"
        assert conv.messages[1].role == "user"
        assert conv.messages[2].role == "assistant"

    def test_add_tool_messages(self):
        conv = Conversation()
        conv.add("", Role.ASSISTANT, tool_calls=[{"id": "tc_1", "name": "search"}])
        conv.add("results", Role.TOOL, tool_call_id="tc_1")

        assert conv.message_count == 2
        assert conv.messages[0].tool_calls is not None
        assert conv.messages[1].tool_call_id == "tc_1"

    def test_default_role_is_user(self):
        conv = Conversation()
        conv.add("hello")
        assert conv.messages[0].role == "user"

    def test_token_tracking(self):
        conv = Conversation()
        conv.add("hello")
        initial = conv.token_count
        assert initial > 0

        conv.add("world", Role.ASSISTANT)
        assert conv.token_count > initial

    def test_usage_ratio(self):
        config = Config(max_tokens=100)
        conv = Conversation(config=config)
        assert conv.usage_ratio == 0.0

        conv.add("test")
        assert 0.0 < conv.usage_ratio < 1.0

    def test_usage_ratio_zero_max(self):
        config = Config(max_tokens=0)
        conv = Conversation(config=config)
        assert conv.usage_ratio == 0.0

    def test_needs_compaction(self):
        config = Config(max_tokens=50, compact_threshold=0.5)
        conv = Conversation(config=config)

        # Add enough content to exceed threshold
        conv.add("a" * 200)
        assert conv.needs_compaction()

    def test_no_compaction_needed(self):
        config = Config(max_tokens=100000)
        conv = Conversation(config=config)
        conv.add("short message")
        assert not conv.needs_compaction()

    def test_clear(self):
        conv = Conversation()
        conv.add("hello")
        conv.add("hi", Role.ASSISTANT)
        conv.clear()

        assert conv.message_count == 0
        assert conv.token_count == 0

    def test_replace_messages(self):
        conv = Conversation()
        conv.add("old message")
        conv.add("old response", Role.ASSISTANT)

        new_msgs = [Message(role="user", content="new")]
        conv.replace_messages(new_msgs)

        assert conv.message_count == 1
        assert conv.messages[0].content == "new"
        # Token count should be recalculated
        assert conv.token_count > 0

    def test_get_system_message(self):
        conv = Conversation()
        conv.add("system prompt", Role.SYSTEM)
        conv.add("hello")

        sys_msg = conv.get_system_message()
        assert sys_msg is not None
        assert sys_msg.content == "system prompt"

    def test_get_system_message_none(self):
        conv = Conversation()
        conv.add("hello")
        assert conv.get_system_message() is None

    def test_messages_returns_copy(self):
        conv = Conversation()
        conv.add("hello")
        msgs = conv.messages
        msgs.append(Message(role="user", content="extra"))
        assert conv.message_count == 1  # Original unchanged

    def test_auto_compaction_with_injected_compactor(self):
        """Test that compaction fires automatically when compactor is set."""
        from llm_kelt.conversation import SlidingWindowCompactor

        config = Config(max_tokens=50, compact_threshold=0.5, min_recent_messages=1)
        conv = Conversation(config=config, compactor=SlidingWindowCompactor())

        conv.add("first message")
        conv.add("second message", Role.ASSISTANT)
        conv.add("a" * 200)  # Should trigger compaction

        # Should have been compacted — only recent messages preserved
        assert conv.message_count <= 2

    def test_no_auto_compaction_without_compactor(self):
        config = Config(max_tokens=50, compact_threshold=0.5)
        conv = Conversation(config=config)

        conv.add("a" * 200)
        # needs_compaction is True, but no compactor means no auto-compact
        assert conv.needs_compaction()
        assert conv.message_count == 1


class TestSplitForCompaction:
    """Tests for Conversation.split_for_compaction()."""

    def test_split_basic(self):
        config = Config(min_recent_messages=2)
        conv = Conversation(config=config)
        conv.add("msg1")
        conv.add("resp1", Role.ASSISTANT)
        conv.add("msg2")
        conv.add("resp2", Role.ASSISTANT)

        to_compact, preserved = conv.split_for_compaction()
        assert len(to_compact) == 2
        assert len(preserved) == 2

    def test_split_preserves_system(self):
        config = Config(min_recent_messages=2, preserve_system=True)
        conv = Conversation(config=config)
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

    def test_split_not_enough_messages(self):
        config = Config(min_recent_messages=4)
        conv = Conversation(config=config)
        conv.add("msg1")
        conv.add("resp1", Role.ASSISTANT)

        to_compact, preserved = conv.split_for_compaction()
        assert len(to_compact) == 0
        assert len(preserved) == 2

    def test_split_no_preserve_system(self):
        config = Config(min_recent_messages=2, preserve_system=False)
        conv = Conversation(config=config)
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
