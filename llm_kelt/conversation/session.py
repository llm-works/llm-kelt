"""Conversation session management.

Manages conversation history with context window awareness. Tracks messages,
estimates token usage, and signals when compaction is needed. Does not perform
compaction itself — that's delegated to a Compactor.

Adapted from llm-gent's core/conv/conversation.py for use as a standalone
primitive without agent-framework dependencies.
"""

from __future__ import annotations

from typing import Any

from appinfra import FieldDict

from .compaction.base import Compactor
from .tokens import estimate_message_tokens
from .types import Message, Role


class Config(FieldDict):
    """Configuration for conversation management.

    Attributes:
        max_tokens: Maximum tokens before compaction is required.
        compact_threshold: Trigger compaction when usage exceeds this fraction (0.0-1.0).
        preserve_system: Always preserve the system message during compaction.
        min_recent_messages: Minimum recent messages to preserve during compaction.
    """

    max_tokens: int = 32000
    compact_threshold: float = 0.8
    preserve_system: bool = True
    min_recent_messages: int = 4


class Conversation:
    """Manages conversation history with context window awareness.

    Tracks messages, estimates token usage, and signals when compaction is needed.
    It does not perform compaction itself — that's delegated to a Compactor.

    Example::

        from llm_kelt.conversation import Role

        # Auto-compaction via injected compactor
        conversation = Conversation(
            config=Config(max_tokens=32000),
            compactor=SlidingWindowCompactor(),
        )
        conversation.add("You are a helpful assistant.", Role.SYSTEM)
        conversation.add("Hello!")  # defaults to Role.USER
        conversation.add("Hi there!", Role.ASSISTANT)
        # Compaction happens automatically when threshold is exceeded

        messages = conversation.messages  # For LLM call
    """

    def __init__(
        self,
        config: Config | None = None,
        compactor: Compactor | None = None,
    ) -> None:
        """Initialize conversation.

        Args:
            config: Conversation configuration. Uses defaults if None.
            compactor: Optional compactor for automatic compaction on add().
        """
        self.config = config or Config()
        self.compactor = compactor
        self._messages: list[Message] = []
        self._token_count: int = 0

    def add(
        self,
        content: str,
        role: Role = Role.USER,
        *,
        tool_calls: list[dict[str, Any]] | None = None,
        tool_call_id: str | None = None,
    ) -> None:
        """Add a message to the conversation.

        If a compactor is configured and the conversation exceeds the
        compaction threshold after adding, compaction runs automatically.

        Args:
            content: Message text content.
            role: Message role (default: Role.USER).
            tool_calls: Tool calls requested by assistant (assistant messages only).
            tool_call_id: ID of the tool call this responds to (tool messages only).
        """
        msg = Message(
            role=role,
            content=content,
            tool_calls=tool_calls,
            tool_call_id=tool_call_id,
        )
        tokens = estimate_message_tokens(msg.role, msg.content, msg.tool_calls)
        self._messages.append(msg)
        self._token_count += tokens

        if self.compactor is not None and self.needs_compaction():
            self.compactor.compact(self)

    @property
    def messages(self) -> list[Message]:
        """Get all messages (copy).

        Returns:
            Copy of message list.
        """
        return list(self._messages)

    def messages_as_dicts(self) -> list[dict[str, Any]]:
        """Get messages as plain dicts, omitting None-valued fields.

        Suitable for passing directly to LLM client APIs that reject
        unexpected null fields (e.g. ``tool_calls: null`` on user messages).
        """
        return [{k: v for k, v in dict(m).items() if v is not None} for m in self._messages]

    @property
    def message_count(self) -> int:
        """Number of messages in conversation."""
        return len(self._messages)

    @property
    def token_count(self) -> int:
        """Estimated token count."""
        return self._token_count

    @property
    def token_limit(self) -> int:
        """Maximum token limit from config."""
        return self.config.max_tokens

    @property
    def usage_ratio(self) -> float:
        """Current usage as fraction of limit (0.0-1.0)."""
        if self.config.max_tokens <= 0:
            return 0.0
        return self._token_count / self.config.max_tokens

    def needs_compaction(self) -> bool:
        """Check if conversation needs compaction.

        Returns:
            True if token usage exceeds compact_threshold.
        """
        return self.usage_ratio >= self.config.compact_threshold

    def clear(self) -> None:
        """Clear all messages and reset token count."""
        self._messages.clear()
        self._token_count = 0

    def replace_messages(self, messages: list[Message]) -> None:
        """Replace all messages (used by compactors).

        Args:
            messages: New message list.
        """
        self._messages = list(messages)
        self._token_count = sum(
            estimate_message_tokens(m.role, m.content, m.tool_calls) for m in self._messages
        )

    def get_system_message(self) -> Message | None:
        """Get the system message if present.

        Returns:
            System message or None.
        """
        for msg in self._messages:
            if msg.role == Role.SYSTEM:
                return msg
        return None

    def split_for_compaction(self) -> tuple[list[Message], list[Message]]:
        """Split messages into compactable and preserved portions.

        Preserved messages include:
        - System message (if preserve_system is True)
        - Last min_recent_messages messages

        Returns:
            Tuple of (messages_to_compact, messages_to_preserve).
        """
        preserve_count = self.config.min_recent_messages
        system_msg = self.get_system_message() if self.config.preserve_system else None

        # When preserving system separately, exclude it from the split pool.
        # Otherwise treat it as a regular message in the pool.
        if system_msg is not None:
            pool = [m for m in self._messages if m.role != Role.SYSTEM]
        else:
            pool = list(self._messages)

        if len(pool) <= preserve_count:
            return [], list(self._messages)

        to_compact = pool[:-preserve_count]
        to_preserve = pool[-preserve_count:]

        if system_msg is not None:
            to_preserve = [system_msg] + to_preserve

        return to_compact, to_preserve

    def __len__(self) -> int:
        """Number of messages."""
        return len(self._messages)
