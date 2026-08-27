# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-kelt Authors

"""Unit tests for conversation types, session, and token estimation."""

from __future__ import annotations

import dataclasses
from unittest.mock import MagicMock

import pytest

from llm_kelt.conversation import (
    AsyncCompactor,
    Compactor,
    Config,
    ContextOverflowError,
    Conversation,
    Message,
    Role,
    ToolCall,
    estimate_message_tokens,
    estimate_tokens,
)
from llm_kelt.conversation.compaction import (
    CompactionContext,
    CompactionGuard,
    CompactionGuardError,
    SummarizingCompactor,
    token_reduction,
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

    def test_tokenizer_callable(self):
        """Test that custom tokenizer is used when provided."""

        def fake_tokenizer(text: str) -> int:
            return 999  # Always return 999 tokens

        assert estimate_tokens("hello", tokenizer=fake_tokenizer) == 999
        assert estimate_message_tokens("user", "hello", tokenizer=fake_tokenizer) == 999 + 4

    def test_tokenizer_empty_string(self):
        """Empty string returns 0 even with tokenizer."""

        def fake_tokenizer(text: str) -> int:
            return 999

        assert estimate_tokens("", tokenizer=fake_tokenizer) == 0


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

    def test_tokenizer_config(self, lg):
        """Test that custom tokenizer in config is used for token counting."""

        def precise_tokenizer(text: str) -> int:
            return len(text.split())  # Word count as tokens

        config = Config(tokenizer=precise_tokenizer)
        conv = Conversation(lg, config=config)
        conv.add("one two three")  # 3 words + 4 overhead = 7 tokens

        assert conv.token_count == 7

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
        # max_tokens must be large enough to hold the message
        config = Config(max_tokens=100, compact_threshold=0.5)
        conv = Conversation(lg, config=config)

        # Add enough content to exceed threshold (54 tokens > 50% of 100)
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

        # max_tokens must accommodate preserved messages after compaction
        config = Config(max_tokens=100, compact_threshold=0.5, min_recent_messages=1)
        conv = Conversation(lg, config=config, compactor=SlidingWindowCompactor())

        conv.add("first message")
        conv.add("second message", Role.ASSISTANT)
        conv.add("a" * 200)  # Should trigger compaction

        # Should have been compacted — only recent messages preserved
        assert conv.message_count <= 2

    def test_no_auto_compaction_without_compactor(self, lg):
        # Without a compactor, messages that fit under max_tokens are allowed
        # but no automatic compaction happens
        config = Config(max_tokens=100, compact_threshold=0.5)
        conv = Conversation(lg, config=config)

        conv.add("a" * 200)  # 54 tokens, under max_tokens but over threshold
        # needs_compaction is True, but no compactor means no auto-compact
        assert conv.needs_compaction()
        assert conv.message_count == 1

    def test_exceeds_max_tokens_without_compactor_raises(self, lg):
        """Without a compactor, exceeding max_tokens raises ContextOverflowError."""
        config = Config(max_tokens=20)  # Very small limit
        conv = Conversation(lg, config=config)

        with pytest.raises(ContextOverflowError) as exc_info:
            conv.add("a" * 200)  # 54 tokens, exceeds max_tokens

        assert exc_info.value.max_tokens == 20
        assert conv.message_count == 0  # Message was NOT added

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

        # max_tokens must accommodate preserved messages after compaction
        config = Config(max_tokens=100, compact_threshold=0.5)
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

        # max_tokens must accommodate preserved messages after compaction
        config = Config(max_tokens=100, compact_threshold=0.5)
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

    def test_context_overflow_raises_when_compaction_insufficient(self, lg):
        """Test that ContextOverflowError is raised when compaction can't meet max_tokens."""

        class NoOpCompactor(Compactor):
            def compact(self, conversation: Conversation) -> None:
                pass  # Doesn't actually reduce anything

        # max_tokens is too small to hold even the preserved messages
        config = Config(max_tokens=10, compact_threshold=0.5)
        conv = Conversation(lg, config=config, compactor=NoOpCompactor())

        conv.add("short")  # Under threshold, no compaction yet

        with pytest.raises(ContextOverflowError) as exc_info:
            conv.add("a" * 200)  # Triggers compaction, but NoOpCompactor doesn't reduce

        assert exc_info.value.max_tokens == 10
        assert exc_info.value.token_count > 10

    async def test_context_overflow_async(self, lg):
        """Test that ContextOverflowError is raised in async path."""

        class NoOpAsyncCompactor(AsyncCompactor):
            async def compact(self, conversation: Conversation) -> None:
                pass

        config = Config(max_tokens=10, compact_threshold=0.5)
        conv = Conversation(lg, config=config, compactor=NoOpAsyncCompactor())

        await conv.append_async(Message(role="user", content="short"))

        with pytest.raises(ContextOverflowError):
            await conv.append_async(Message(role="user", content="a" * 200))


# =============================================================================
# SummarizingCompactor
# =============================================================================


class MockChatResponse:
    """Mock response from LLM client."""

    def __init__(self, content: str):
        self.content = content


class MockChatClient:
    """Mock LLM client for testing."""

    def __init__(self, responses: list[str] | None = None):
        self.responses = responses or ["Summary of conversation."]
        self.call_count = 0
        self.last_messages: list[dict] | None = None

    async def chat_async(self, messages: list[dict], **kwargs) -> MockChatResponse:
        self.last_messages = messages
        response = self.responses[min(self.call_count, len(self.responses) - 1)]
        self.call_count += 1
        return MockChatResponse(response)


class TestSummarizingCompactor:
    """Tests for SummarizingCompactor."""

    async def test_basic_compaction(self, lg):
        """Test basic summarization without guards."""
        client = MockChatClient(responses=["This is a summary."])
        compactor = SummarizingCompactor(client=client)

        config = Config(max_tokens=200, compact_threshold=0.5, min_recent_messages=1)
        conv = Conversation(lg, config=config, compactor=compactor)

        conv.add("system prompt", Role.SYSTEM)
        conv.add("first message")
        conv.add("first response", Role.ASSISTANT)
        conv.add("second message")

        # Trigger compaction via append_async
        await conv.append_async(Message(role="assistant", content="a" * 500))

        assert client.call_count == 1
        # Should have system + summary + recent messages
        assert any("[Previous conversation summary]" in m.content for m in conv.messages)

    async def test_guard_passes(self, lg):
        """Test that guards are validated and pass."""
        client = MockChatClient(responses=["Short summary."])

        def always_pass(ctx: CompactionContext) -> str | None:
            return None  # No error = pass

        guard = CompactionGuard(
            validator=always_pass,
            retry_instruction="Make it better.",
            name="test_guard",
        )
        compactor = SummarizingCompactor(client=client, guards=[guard])

        config = Config(max_tokens=200, compact_threshold=0.5, min_recent_messages=1)
        conv = Conversation(lg, config=config, compactor=compactor)

        conv.add("first message")
        conv.add("first response", Role.ASSISTANT)

        await conv.append_async(Message(role="user", content="a" * 500))

        assert client.call_count == 1  # No retries needed

    async def test_guard_retry_then_pass(self, lg):
        """Test that guard failure triggers retry with feedback."""
        # First response fails, second passes
        client = MockChatClient(responses=["Too long summary.", "Short."])

        attempt_count = [0]

        def fail_first_attempt(ctx: CompactionContext) -> str | None:
            attempt_count[0] += 1
            if attempt_count[0] == 1:
                return "Summary too long"
            return None

        guard = CompactionGuard(
            validator=fail_first_attempt,
            retry_instruction="Make it shorter.",
            max_retries=2,
            name="length_check",
        )
        compactor = SummarizingCompactor(client=client, guards=[guard])

        config = Config(max_tokens=200, compact_threshold=0.5, min_recent_messages=1)
        conv = Conversation(lg, config=config, compactor=compactor)

        conv.add("first message")
        conv.add("first response", Role.ASSISTANT)

        await conv.append_async(Message(role="user", content="a" * 500))

        assert client.call_count == 2  # Initial + 1 retry
        # Second call should include feedback
        assert "Make it shorter." in client.last_messages[1]["content"]

    async def test_guard_exhausts_retries(self, lg):
        """Test that CompactionGuardError is raised when retries exhausted."""
        client = MockChatClient(responses=["Always fails."] * 10)

        def always_fail(ctx: CompactionContext) -> str | None:
            return "This always fails"

        guard = CompactionGuard(
            validator=always_fail,
            retry_instruction="Try again.",
            max_retries=2,
            name="failing_guard",
        )
        compactor = SummarizingCompactor(client=client, guards=[guard])

        config = Config(max_tokens=200, compact_threshold=0.5, min_recent_messages=1)
        conv = Conversation(lg, config=config, compactor=compactor)

        conv.add("first message")
        conv.add("first response", Role.ASSISTANT)

        with pytest.raises(CompactionGuardError) as exc_info:
            await conv.append_async(Message(role="user", content="a" * 500))

        assert exc_info.value.attempts == 3  # 1 initial + 2 retries
        assert "This always fails" in exc_info.value.error
        assert client.call_count == 3

    async def test_tokenizer_from_conversation_config(self, lg):
        """Test that compactor uses conversation's tokenizer when not provided."""
        client = MockChatClient(responses=["Summary."])

        def check_token_count(ctx: CompactionContext) -> str | None:
            # With word tokenizer, "Summary." = 1 word + overhead
            # Preserved messages will also be counted
            # This just verifies the guard can access token counts
            if ctx.after_tokens == 0:
                return "Token count should not be zero"
            return None

        guard = CompactionGuard(
            validator=check_token_count,
            retry_instruction="Fix it.",
            name="token_check",
        )

        def word_tokenizer(text: str) -> int:
            return len(text.split())

        compactor = SummarizingCompactor(client=client, guards=[guard])
        config = Config(
            max_tokens=500,
            compact_threshold=0.3,
            min_recent_messages=1,
            tokenizer=word_tokenizer,
        )
        conv = Conversation(lg, config=config, compactor=compactor)

        conv.add("first message here")
        conv.add("first response here", Role.ASSISTANT)

        await conv.append_async(Message(role="user", content="a " * 200))

        assert client.call_count == 1  # Guard passed

    async def test_after_tokens_includes_preserved(self, lg):
        """Test that after_tokens correctly includes preserved messages."""
        client = MockChatClient(responses=["Short."])

        captured_ctx: list[CompactionContext] = []

        def capture_context(ctx: CompactionContext) -> str | None:
            captured_ctx.append(ctx)
            return None

        guard = CompactionGuard(
            validator=capture_context,
            retry_instruction="N/A",
            name="capture",
        )
        compactor = SummarizingCompactor(client=client, guards=[guard])

        config = Config(max_tokens=500, compact_threshold=0.3, min_recent_messages=2)
        conv = Conversation(lg, config=config, compactor=compactor)

        conv.add("system prompt", Role.SYSTEM)
        conv.add("message one")
        conv.add("response one", Role.ASSISTANT)
        conv.add("message two")
        conv.add("response two", Role.ASSISTANT)

        await conv.append_async(Message(role="user", content="a " * 300))

        assert len(captured_ctx) == 1
        ctx = captured_ctx[0]

        # after_messages should include system + summary + preserved non-system
        assert len(ctx.after_messages) >= 2
        # after_tokens should be > just the summary tokens
        summary_only_tokens = estimate_message_tokens(
            "user", "[Previous conversation summary]\nShort.\n[End summary]"
        )
        assert ctx.after_tokens > summary_only_tokens

    async def test_after_messages_populated(self, lg):
        """Test that after_messages is correctly populated for guards."""
        client = MockChatClient(responses=["The summary."])

        captured_ctx: list[CompactionContext] = []

        def capture_context(ctx: CompactionContext) -> str | None:
            captured_ctx.append(ctx)
            return None

        guard = CompactionGuard(
            validator=capture_context,
            retry_instruction="N/A",
            name="capture",
        )
        compactor = SummarizingCompactor(client=client, guards=[guard])

        config = Config(max_tokens=300, compact_threshold=0.3, min_recent_messages=1)
        conv = Conversation(lg, config=config, compactor=compactor)

        conv.add("system prompt", Role.SYSTEM)
        conv.add("old message")
        conv.add("old response", Role.ASSISTANT)

        await conv.append_async(Message(role="user", content="a " * 200))

        assert len(captured_ctx) == 1
        ctx = captured_ctx[0]

        # after_messages should not be empty
        assert len(ctx.after_messages) > 0
        # Should contain the summary message
        assert any("The summary." in m.content for m in ctx.after_messages)


class TestConversationSerialization:
    """Tests for Conversation.to_dict() / from_dict()."""

    def test_to_dict_basic(self, lg):
        """Test basic serialization."""
        conv = Conversation(lg)
        conv.add("You are helpful.", Role.SYSTEM)
        conv.add("Hello")
        conv.add("Hi there!", Role.ASSISTANT)

        d = conv.to_dict()
        assert "messages" in d
        assert "token_count" in d
        assert len(d["messages"]) == 3
        assert d["token_count"] == conv.token_count

    def test_from_dict_basic(self, lg):
        """Test basic deserialization."""
        conv = Conversation(lg)
        conv.add("Hello")
        conv.add("Hi!", Role.ASSISTANT)

        d = conv.to_dict()
        restored = Conversation.from_dict(d, lg)

        assert restored.message_count == 2
        assert restored.token_count == conv.token_count
        assert restored.messages[0].content == "Hello"
        assert restored.messages[1].role == "assistant"

    def test_roundtrip_with_tool_calls(self, lg):
        """Test serialization preserves tool calls."""
        conv = Conversation(lg)
        conv.add(
            "",
            Role.ASSISTANT,
            tool_calls=[ToolCall(id="tc_1", name="search", arguments={"q": "test"})],
        )
        conv.add("results", Role.TOOL, tool_call_id="tc_1")

        d = conv.to_dict()
        restored = Conversation.from_dict(d, lg)

        assert restored.messages[0].tool_calls is not None
        assert restored.messages[0].tool_calls[0].name == "search"
        assert restored.messages[0].tool_calls[0].arguments == {"q": "test"}
        assert restored.messages[1].tool_call_id == "tc_1"

    def test_from_dict_with_config(self, lg):
        """Test that config is applied on restore."""
        conv = Conversation(lg)
        conv.add("test")

        d = conv.to_dict()
        config = Config(max_tokens=5000, compact_threshold=0.6)
        restored = Conversation.from_dict(d, lg, config=config)

        assert restored.config.max_tokens == 5000
        assert restored.config.compact_threshold == 0.6

    def test_from_dict_with_compactor(self, lg):
        """Test that compactor is applied on restore."""
        from llm_kelt.conversation import SlidingWindowCompactor

        conv = Conversation(lg)
        conv.add("test")

        d = conv.to_dict()
        compactor = SlidingWindowCompactor()
        restored = Conversation.from_dict(d, lg, compactor=compactor)

        assert restored.compactor is compactor

    def test_json_serializable(self, lg):
        """Test that to_dict output is JSON-serializable."""
        import json

        conv = Conversation(lg)
        conv.add("system", Role.SYSTEM)
        conv.add("Hello")
        conv.add(
            "Let me search",
            Role.ASSISTANT,
            tool_calls=[ToolCall(id="tc_1", name="search", arguments={"q": "test"})],
        )
        conv.add("results", Role.TOOL, tool_call_id="tc_1")

        d = conv.to_dict()
        json_str = json.dumps(d)  # Should not raise
        parsed = json.loads(json_str)

        # Verify roundtrip through JSON
        restored = Conversation.from_dict(parsed, lg)
        assert restored.message_count == 4

    def test_empty_conversation_roundtrip(self, lg):
        """Test serialization of empty conversation."""
        conv = Conversation(lg)
        d = conv.to_dict()
        restored = Conversation.from_dict(d, lg)

        assert restored.message_count == 0
        assert restored.token_count == 0

    def test_from_dict_recalculates_tokens_with_new_tokenizer(self, lg):
        """Test that token count is recalculated using restored config's tokenizer."""
        # Create conversation with default tokenizer
        conv = Conversation(lg)
        conv.add("Hello world")
        original_tokens = conv.token_count

        d = conv.to_dict()

        # Restore with a custom tokenizer that counts differently (1 token per char)
        def char_tokenizer(text: str) -> int:
            return len(text)

        config = Config(tokenizer=char_tokenizer)
        restored = Conversation.from_dict(d, lg, config=config)

        # Token count should be recalculated, not copied from serialized data
        assert restored.token_count != original_tokens
        # With char_tokenizer, "Hello world" = 11 chars, plus role overhead
        assert restored.token_count > 0

    def test_from_dict_validates_input(self, lg):
        """Test that from_dict raises ValueError on malformed input."""
        import pytest

        with pytest.raises(ValueError, match="Expected dict"):
            Conversation.from_dict("not a dict", lg)

        with pytest.raises(ValueError, match="Missing required key 'messages'"):
            Conversation.from_dict({}, lg)

        with pytest.raises(ValueError, match="Expected 'messages' to be a list"):
            Conversation.from_dict({"messages": "not a list"}, lg)

        with pytest.raises(ValueError, match=r"Expected messages\[1\] to be a dict"):
            Conversation.from_dict({"messages": [{}, "not a dict"]}, lg)


class TestTokenReductionGuard:
    """Tests for the token_reduction pre-built guard."""

    def test_passes_when_reduction_sufficient(self):
        """Test guard passes when reduction ratio meets minimum."""
        guard = token_reduction(min_ratio=0.3)

        ctx = CompactionContext(
            before_messages=[],
            after_messages=[],
            before_tokens=100,
            after_tokens=60,  # 40% reduction
            summary="test",
            summary_tokens=50,
            attempt=0,
        )

        result = guard.validator(ctx)
        assert result is None  # Pass

    def test_fails_when_reduction_insufficient(self):
        """Test guard fails when reduction ratio is below minimum."""
        guard = token_reduction(min_ratio=0.3)

        ctx = CompactionContext(
            before_messages=[],
            after_messages=[],
            before_tokens=100,
            after_tokens=90,  # Only 10% reduction
            summary="test",
            summary_tokens=80,
            attempt=0,
        )

        result = guard.validator(ctx)
        assert result is not None
        assert "10.0%" in result
        assert "30.0%" in result
