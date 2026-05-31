"""Unit tests for conversation compaction strategies."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from llm_kelt.conversation import (
    Config,
    Conversation,
    Message,
    Role,
    SlidingWindowCompactor,
    TieredCompactor,
    ToolCall,
)
from llm_kelt.conversation.compaction import DEFAULT_TRIMMABLE_TOOLS, AsyncTieredCompactor


class TestSlidingWindowCompactor:
    """Tests for SlidingWindowCompactor."""

    def test_compact_drops_old_messages(self, lg):
        config = Config(min_recent_messages=2)
        conv = Conversation(lg, config=config)
        conv.add("old1")
        conv.add("old_resp1", Role.ASSISTANT)
        conv.add("recent1")
        conv.add("recent_resp1", Role.ASSISTANT)

        compactor = SlidingWindowCompactor()
        compactor.compact(conv)

        assert conv.message_count == 2
        assert conv.messages[0].content == "recent1"
        assert conv.messages[1].content == "recent_resp1"

    def test_compact_preserves_system(self, lg):
        config = Config(min_recent_messages=2, preserve_system=True)
        conv = Conversation(lg, config=config)
        conv.add("system prompt", Role.SYSTEM)
        conv.add("old1")
        conv.add("old_resp1", Role.ASSISTANT)
        conv.add("recent1")
        conv.add("recent_resp1", Role.ASSISTANT)

        compactor = SlidingWindowCompactor()
        compactor.compact(conv)

        assert conv.message_count == 3
        assert conv.messages[0].role == "system"
        assert conv.messages[0].content == "system prompt"
        assert conv.messages[1].content == "recent1"

    def test_compact_nothing_to_compact(self, lg):
        config = Config(min_recent_messages=4)
        conv = Conversation(lg, config=config)
        conv.add("msg1")
        conv.add("resp1", Role.ASSISTANT)

        compactor = SlidingWindowCompactor()
        compactor.compact(conv)

        # Nothing changed — not enough messages to compact
        assert conv.message_count == 2

    def test_compact_updates_token_count(self, lg):
        config = Config(min_recent_messages=1)
        conv = Conversation(lg, config=config)
        conv.add("a" * 200)
        conv.add("short")

        tokens_before = conv.token_count

        compactor = SlidingWindowCompactor()
        compactor.compact(conv)

        assert conv.token_count < tokens_before
        assert conv.message_count == 1

    def test_compact_empty_conversation(self, lg):
        conv = Conversation(lg)
        compactor = SlidingWindowCompactor()
        compactor.compact(conv)
        assert conv.message_count == 0


class TestTieredCompactor:
    """Tests for TieredCompactor."""

    def _make_mock_client(self, summary_content: str = "Summary of conversation."):
        """Create a mock LLM client."""
        client = MagicMock()
        response = MagicMock()
        response.content = summary_content
        client.chat.return_value = response
        return client

    def test_phase1_trims_large_tool_results(self, lg):
        """Phase 1 should trim large tool outputs from trimmable tools."""
        config = Config(max_tokens=100000, compact_threshold=0.8)
        conv = Conversation(lg, config=config)

        # Add assistant with tool call
        conv.append(
            Message(
                role=Role.ASSISTANT,
                content="Searching...",
                tool_calls=[ToolCall(id="tc_1", name="web_search", arguments={})],
            )
        )
        # Add large tool result
        large_content = "x" * 5000
        conv.append(Message(role=Role.TOOL, content=large_content, tool_call_id="tc_1"))

        client = self._make_mock_client()
        compactor = TieredCompactor(client, trim_threshold=2000)
        compactor.compact(conv)

        # Tool result should be trimmed
        tool_msg = [m for m in conv.messages if m.role == Role.TOOL][0]
        assert len(tool_msg.content) < 5000
        assert "trimmed" in tool_msg.content.lower()
        assert tool_msg.tool_call_id == "tc_1"

        # LLM should NOT be called (trimming was sufficient)
        client.chat.assert_not_called()

    def test_phase1_preserves_small_tool_results(self, lg):
        """Phase 1 should not trim small tool outputs."""
        config = Config(max_tokens=100000)
        conv = Conversation(lg, config=config)

        conv.append(
            Message(
                role=Role.ASSISTANT,
                content="Searching...",
                tool_calls=[ToolCall(id="tc_1", name="web_search", arguments={})],
            )
        )
        small_content = "Small result"
        conv.append(Message(role=Role.TOOL, content=small_content, tool_call_id="tc_1"))

        client = self._make_mock_client()
        compactor = TieredCompactor(client, trim_threshold=2000)
        compactor.compact(conv)

        tool_msg = [m for m in conv.messages if m.role == Role.TOOL][0]
        assert tool_msg.content == small_content

    def test_phase1_ignores_non_trimmable_tools(self, lg):
        """Phase 1 should not trim tools not in trimmable_tools set."""
        config = Config(max_tokens=100000)
        conv = Conversation(lg, config=config)

        conv.append(
            Message(
                role=Role.ASSISTANT,
                content="Running...",
                tool_calls=[ToolCall(id="tc_1", name="custom_tool", arguments={})],
            )
        )
        large_content = "x" * 5000
        conv.append(Message(role=Role.TOOL, content=large_content, tool_call_id="tc_1"))

        client = self._make_mock_client()
        compactor = TieredCompactor(client, trim_threshold=2000)
        compactor.compact(conv)

        tool_msg = [m for m in conv.messages if m.role == Role.TOOL][0]
        assert tool_msg.content == large_content  # Not trimmed

    def test_phase2_summarizes_when_trimming_insufficient(self, lg):
        """Phase 2 should summarize if still over threshold after trimming."""
        # Use low max_tokens with low compact_threshold to force compaction
        config = Config(max_tokens=300, compact_threshold=0.5, min_recent_messages=2)
        conv = Conversation(lg, config=config)

        # Add enough messages to exceed 50% threshold (~150 tokens)
        for i in range(8):
            conv.add(f"User message number {i} with content")
            conv.add(f"Response {i} here", Role.ASSISTANT)

        # Verify we're in compaction territory
        assert conv.needs_compaction(), f"Expected compaction at {conv.usage_ratio:.1%}"

        client = self._make_mock_client("Summarized conversation content.")
        compactor = TieredCompactor(client)
        compactor.compact(conv)

        # LLM should be called for summarization
        client.chat.assert_called_once()

        # Should have summary message
        messages = conv.messages
        summary_msg = [m for m in messages if "summary" in (m.content or "").lower()]
        assert len(summary_msg) > 0

    def test_custom_trimmable_tools(self, lg):
        """Should respect custom trimmable_tools set."""
        config = Config(max_tokens=100000)
        conv = Conversation(lg, config=config)

        conv.append(
            Message(
                role=Role.ASSISTANT,
                content="Running...",
                tool_calls=[ToolCall(id="tc_1", name="my_custom_tool", arguments={})],
            )
        )
        large_content = "x" * 5000
        conv.append(Message(role=Role.TOOL, content=large_content, tool_call_id="tc_1"))

        client = self._make_mock_client()
        compactor = TieredCompactor(
            client,
            trim_threshold=2000,
            trimmable_tools={"my_custom_tool"},
        )
        compactor.compact(conv)

        tool_msg = [m for m in conv.messages if m.role == Role.TOOL][0]
        assert "trimmed" in tool_msg.content.lower()

    def test_preserves_system_message(self, lg):
        """System messages should be preserved through compaction."""
        config = Config(
            max_tokens=500, compact_threshold=0.5, min_recent_messages=2, preserve_system=True
        )
        conv = Conversation(lg, config=config)
        conv.add("System prompt", Role.SYSTEM)

        for i in range(10):
            conv.add(f"User message {i}")
            conv.add(f"Response {i}", Role.ASSISTANT)

        client = self._make_mock_client()
        compactor = TieredCompactor(client)
        compactor.compact(conv)

        assert conv.messages[0].role == Role.SYSTEM
        assert conv.messages[0].content == "System prompt"

    def test_default_trimmable_tools_exported(self):
        """DEFAULT_TRIMMABLE_TOOLS should be exported and contain expected tools."""
        assert "web_search" in DEFAULT_TRIMMABLE_TOOLS
        assert "web_fetch" in DEFAULT_TRIMMABLE_TOOLS
        assert isinstance(DEFAULT_TRIMMABLE_TOOLS, frozenset)


class TestAsyncTieredCompactor:
    """Tests for AsyncTieredCompactor."""

    def _make_mock_async_client(self, summary_content: str = "Summary."):
        """Create a mock async LLM client."""
        client = MagicMock()
        response = MagicMock()
        response.content = summary_content

        async def mock_chat_async(*args, **kwargs):
            return response

        client.chat_async = mock_chat_async
        return client

    @pytest.mark.asyncio
    async def test_async_phase1_trims(self, lg):
        """Async compactor should trim in phase 1."""
        config = Config(max_tokens=100000)
        conv = Conversation(lg, config=config)

        conv.append(
            Message(
                role=Role.ASSISTANT,
                content="Fetching...",
                tool_calls=[ToolCall(id="tc_1", name="web_fetch", arguments={})],
            )
        )
        conv.append(Message(role=Role.TOOL, content="x" * 5000, tool_call_id="tc_1"))

        client = self._make_mock_async_client()
        compactor = AsyncTieredCompactor(client, trim_threshold=2000)
        await compactor.compact(conv)

        tool_msg = [m for m in conv.messages if m.role == Role.TOOL][0]
        assert "trimmed" in tool_msg.content.lower()

    @pytest.mark.asyncio
    async def test_async_phase2_summarizes(self, lg):
        """Async compactor should use chat_async for phase 2."""
        config = Config(max_tokens=500, compact_threshold=0.5, min_recent_messages=2)
        conv = Conversation(lg, config=config)

        for i in range(10):
            conv.add(f"Message {i} content")
            conv.add(f"Response {i}", Role.ASSISTANT)

        client = self._make_mock_async_client("Async summary.")
        compactor = AsyncTieredCompactor(client)
        await compactor.compact(conv)

        messages = conv.messages
        summary_msg = [m for m in messages if "summary" in (m.content or "").lower()]
        assert len(summary_msg) > 0


# --- Bug-regression tests ------------------------------------------------


def _no_orphan_tool_call_ids(messages: list[Message]) -> bool:
    """True if every tool_call_id has a matching prior assistant.tool_calls entry."""
    seen: set[str] = set()
    for msg in messages:
        if msg.tool_calls:
            for tc in msg.tool_calls:
                seen.add(tc.id)
        if msg.role == Role.TOOL and msg.tool_call_id not in seen:
            return False
    return True


class TestSplitForCompactionPairing:
    """Bug 1: split must never orphan tool_call_ids."""

    def test_cut_between_assistant_and_tool_walks_back(self, lg):
        config = Config(min_recent_messages=3)
        conv = Conversation(lg, config=config)
        conv.add("first user msg")
        conv.add(
            "calling tool",
            Role.ASSISTANT,
            tool_calls=[ToolCall(id="tc_x", name="web_fetch", arguments={})],
        )
        conv.add("tool result", Role.TOOL, tool_call_id="tc_x")
        conv.add("follow-up", Role.USER)
        conv.add("done", Role.ASSISTANT)

        to_compact, preserved = conv.split_for_compaction()

        assert _no_orphan_tool_call_ids(preserved), preserved
        assert len(to_compact) == 1
        assert to_compact[0].content == "first user msg"
        roles = [m.role for m in preserved]
        assert Role.ASSISTANT in roles
        assert Role.TOOL in roles

    def test_cut_through_multi_tool_group_walks_past_all_tools(self, lg):
        config = Config(min_recent_messages=3)
        conv = Conversation(lg, config=config)
        conv.add("user1")
        conv.add(
            "parallel tools",
            Role.ASSISTANT,
            tool_calls=[
                ToolCall(id="tc_a", name="web_search", arguments={}),
                ToolCall(id="tc_b", name="web_search", arguments={}),
                ToolCall(id="tc_c", name="web_search", arguments={}),
            ],
        )
        conv.add("a", Role.TOOL, tool_call_id="tc_a")
        conv.add("b", Role.TOOL, tool_call_id="tc_b")
        conv.add("c", Role.TOOL, tool_call_id="tc_c")
        conv.add("final", Role.USER)
        conv.add("answer", Role.ASSISTANT)

        to_compact, preserved = conv.split_for_compaction()

        assert _no_orphan_tool_call_ids(preserved)
        assert to_compact == [conv.messages[0]]
        assert sum(1 for m in preserved if m.role == Role.TOOL) == 3

    def test_cut_outside_tool_group_unchanged(self, lg):
        config = Config(min_recent_messages=2)
        conv = Conversation(lg, config=config)
        conv.add("user1")
        conv.add(
            "tools",
            Role.ASSISTANT,
            tool_calls=[ToolCall(id="t1", name="web_search", arguments={})],
        )
        conv.add("res", Role.TOOL, tool_call_id="t1")
        conv.add("user2")
        conv.add("answer", Role.ASSISTANT)

        to_compact, preserved = conv.split_for_compaction()

        assert _no_orphan_tool_call_ids(preserved)
        assert len(to_compact) == 3
        assert sum(1 for m in to_compact if m.role == Role.TOOL) == 1
        assert sum(1 for m in to_compact if m.tool_calls) == 1

    def test_walkback_consumes_whole_pool_returns_no_compact(self, lg):
        config = Config(min_recent_messages=1)
        conv = Conversation(lg, config=config)
        conv.add("sys", Role.SYSTEM)
        conv.add(
            "tools",
            Role.ASSISTANT,
            tool_calls=[ToolCall(id="t1", name="web_fetch", arguments={})],
        )
        conv.add("tool result content", Role.TOOL, tool_call_id="t1")

        to_compact, preserved = conv.split_for_compaction()

        assert to_compact == []
        assert _no_orphan_tool_call_ids(preserved)

    def test_end_to_end_compaction_never_orphans(self, lg):
        config = Config(max_tokens=500, compact_threshold=0.5, min_recent_messages=3)
        conv = Conversation(lg, config=config)
        for i in range(5):
            conv.add(f"please do thing {i}")
            conv.add(
                f"calling tool {i}",
                Role.ASSISTANT,
                tool_calls=[ToolCall(id=f"tc_{i}", name="web_fetch", arguments={})],
            )
            conv.add(f"tool output {i}", Role.TOOL, tool_call_id=f"tc_{i}")

        client = MagicMock()
        resp = MagicMock()
        resp.content = "summary text"
        resp.finish_reason = "stop"
        client.chat.return_value = resp

        compactor = TieredCompactor(client, trim_threshold=10)
        compactor.compact(conv)

        assert _no_orphan_tool_call_ids(conv.messages), [
            (m.role, m.tool_call_id, [tc.id for tc in (m.tool_calls or [])]) for m in conv.messages
        ]


class TestTruncatedSummaryRejected:
    """Bug 2: finish_reason=length must trigger fallback, never apply."""

    def _make_truncated_client(self, partial: str = "this summary was cut o"):
        client = MagicMock()
        response = MagicMock()
        response.content = partial
        response.finish_reason = "length"
        client.chat.return_value = response
        return client

    def test_truncated_summary_triggers_fallback(self, lg):
        config = Config(max_tokens=400, compact_threshold=0.5, min_recent_messages=2)
        conv = Conversation(lg, config=config)
        for i in range(10):
            conv.add(f"user message number {i}")
            conv.add(f"assistant response {i}", Role.ASSISTANT)

        client = self._make_truncated_client()
        compactor = TieredCompactor(client)
        compactor.compact(conv)

        for m in conv.messages:
            assert "this summary was cut o" not in (m.content or "")
        assert not any("Previous conversation summary" in (m.content or "") for m in conv.messages)
        assert conv.message_count == config.min_recent_messages

    def test_normal_summary_unaffected(self, lg):
        config = Config(max_tokens=400, compact_threshold=0.5, min_recent_messages=2)
        conv = Conversation(lg, config=config)
        for i in range(10):
            conv.add(f"user message number {i}")
            conv.add(f"assistant response {i}", Role.ASSISTANT)

        client = MagicMock()
        resp = MagicMock()
        resp.content = "ok summary"
        resp.finish_reason = "stop"
        client.chat.return_value = resp
        compactor = TieredCompactor(client)
        compactor.compact(conv)

        assert any("Previous conversation summary" in (m.content or "") for m in conv.messages)


class TestSummaryRegressionRejected:
    """Bug 3: a summary that grows the conversation must be rejected."""

    def test_summary_larger_than_input_triggers_fallback(self, lg):
        config = Config(max_tokens=400, compact_threshold=0.5, min_recent_messages=2)
        conv = Conversation(lg, config=config)
        for i in range(10):
            conv.add(f"u{i}")
            conv.add(f"a{i}", Role.ASSISTANT)

        before_tokens = conv.token_count
        bloated_summary = "x " * (before_tokens * 4)

        client = MagicMock()
        resp = MagicMock()
        resp.content = bloated_summary
        resp.finish_reason = "stop"
        client.chat.return_value = resp

        compactor = TieredCompactor(client)
        compactor.compact(conv)

        assert not any(bloated_summary in (m.content or "") for m in conv.messages)
        assert conv.token_count < before_tokens, (
            f"compaction failed to shrink: before={before_tokens} after={conv.token_count}"
        )

    @pytest.mark.asyncio
    async def test_async_summary_regression_triggers_fallback(self, lg):
        config = Config(max_tokens=400, compact_threshold=0.5, min_recent_messages=2)
        conv = Conversation(lg, config=config)
        for i in range(10):
            conv.add(f"u{i}")
            conv.add(f"a{i}", Role.ASSISTANT)

        before_tokens = conv.token_count
        bloated = "x " * (before_tokens * 4)

        client = MagicMock()
        resp = MagicMock()
        resp.content = bloated
        resp.finish_reason = "stop"

        async def mock_chat_async(*args, **kwargs):
            return resp

        client.chat_async = mock_chat_async

        compactor = AsyncTieredCompactor(client)
        await compactor.compact(conv)

        assert conv.token_count < before_tokens
