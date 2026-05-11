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
