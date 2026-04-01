"""Unit tests for conversation compaction strategies."""

from __future__ import annotations

from llm_kelt.conversation import (
    Config,
    Conversation,
    Role,
    SlidingWindowCompactor,
)


class TestSlidingWindowCompactor:
    """Tests for SlidingWindowCompactor."""

    def test_compact_drops_old_messages(self):
        config = Config(min_recent_messages=2)
        conv = Conversation(config=config)
        conv.add("old1")
        conv.add("old_resp1", Role.ASSISTANT)
        conv.add("recent1")
        conv.add("recent_resp1", Role.ASSISTANT)

        compactor = SlidingWindowCompactor()
        compactor.compact(conv)

        assert conv.message_count == 2
        assert conv.messages[0].content == "recent1"
        assert conv.messages[1].content == "recent_resp1"

    def test_compact_preserves_system(self):
        config = Config(min_recent_messages=2, preserve_system=True)
        conv = Conversation(config=config)
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

    def test_compact_nothing_to_compact(self):
        config = Config(min_recent_messages=4)
        conv = Conversation(config=config)
        conv.add("msg1")
        conv.add("resp1", Role.ASSISTANT)

        compactor = SlidingWindowCompactor()
        compactor.compact(conv)

        # Nothing changed — not enough messages to compact
        assert conv.message_count == 2

    def test_compact_updates_token_count(self):
        config = Config(min_recent_messages=1)
        conv = Conversation(config=config)
        conv.add("a" * 200)
        conv.add("short")

        tokens_before = conv.token_count

        compactor = SlidingWindowCompactor()
        compactor.compact(conv)

        assert conv.token_count < tokens_before
        assert conv.message_count == 1

    def test_compact_empty_conversation(self):
        conv = Conversation()
        compactor = SlidingWindowCompactor()
        compactor.compact(conv)
        assert conv.message_count == 0
