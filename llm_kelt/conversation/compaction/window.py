"""Sliding window compaction strategy.

Drops oldest messages, keeping only recent ones. Fast and predictable,
but loses information completely. Best for conversations where older
context is less important.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import Compactor

if TYPE_CHECKING:
    from ..session import Conversation


class SlidingWindowCompactor(Compactor):
    """Compactor that drops oldest messages, keeping only recent ones.

    Uses conversation.split_for_compaction() to determine which messages
    to preserve (system message + N most recent messages).
    """

    def compact(self, conversation: Conversation) -> None:
        """Drop old messages, keeping only recent ones."""
        _, preserved = conversation.split_for_compaction()
        conversation.replace_messages(preserved)
