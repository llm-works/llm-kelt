# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-kelt Authors

"""Conversation session management.

Manages conversation history with context window awareness. Tracks messages,
estimates token usage, and signals when compaction is needed. Does not perform
compaction itself — that's delegated to a Compactor.

Implements saia's ``AsyncConversationLike`` protocol so it can be passed directly
to saia verbs (e.g., ``Complete``) for automatic compaction during tool loops.
"""

from __future__ import annotations

import asyncio
import dataclasses
from typing import Any, Self

from appinfra import FieldDict
from appinfra.log import Logger
from appinfra.time import since, start
from llm_saia import AsyncConversationLike, Message, Role, ToolCall

from ..core.errors import ContextOverflowError
from .compaction.base import AsyncCompactor, Compactor
from .tokens import Tokenizer, estimate_message_tokens


def _adjust_cut_to_tool_group_boundary(pool: list[Message], cut: int) -> int:
    """Walk a compaction cut index backward off any tool-result boundary.

    A tool message at ``pool[cut]`` would land at the start of to_preserve
    while its parent assistant (with matching tool_calls) stayed in
    to_compact — orphaning the tool_call_id. Walking back pulls the tool
    results and their parent into to_preserve together.
    """
    while cut > 0 and pool[cut].role == Role.TOOL:
        cut -= 1
    return cut


class Config(FieldDict):
    """Configuration for conversation management.

    Attributes:
        max_tokens: Hard limit on input tokens. Compaction must achieve this or raise.
        compact_threshold: Trigger compaction when usage exceeds this fraction (0.0-1.0).
        preserve_system: Always preserve the system message during compaction.
        min_recent_messages: Minimum recent messages to preserve during compaction.
        tokenizer: Optional tokenizer for accurate token counting. If None, uses
            char/4 heuristic. Pass a callable that takes text and returns token count.
    """

    max_tokens: int = 32000
    compact_threshold: float = 0.8
    preserve_system: bool = True
    min_recent_messages: int = 4
    tokenizer: Tokenizer | None = None


class Conversation(AsyncConversationLike):
    """Manages conversation history with context window awareness.

    Tracks messages, estimates token usage, and signals when compaction is needed.
    It does not perform compaction itself — that's delegated to a Compactor.

    Implements ``AsyncConversationLike`` so it can be passed to saia verbs::

        from appinfra.log import Logger
        from llm_kelt.conversation import Conversation, Config
        from llm_saia import Complete

        conv = Conversation(lg, config=Config(max_tokens=32000))
        result = await complete("Do the task", conversation=conv)
        # conv now contains the full history with automatic compaction

    Example::

        from appinfra.log import Logger
        from llm_kelt.conversation import Conversation, Config, Role

        conversation = Conversation(
            lg,
            config=Config(max_tokens=32000),
            compactor=SlidingWindowCompactor(),
        )
        conversation.add("You are a helpful assistant.", Role.SYSTEM)
        conversation.add("Hello!")  # defaults to Role.USER
        conversation.add("Hi there!", Role.ASSISTANT)
    """

    def __init__(
        self,
        lg: Logger,
        config: Config | None = None,
        compactor: Compactor | AsyncCompactor | None = None,
    ) -> None:
        """Initialize conversation.

        Args:
            lg: Logger for debug output (compaction events, stats).
            config: Conversation configuration. Uses defaults if None.
            compactor: Optional compactor for automatic compaction on add().
                If an AsyncCompactor is provided, use append_async() instead
                of append() to avoid blocking the event loop.
        """
        self._lg = lg
        self.config = config or Config()
        self.compactor = compactor
        self._messages: list[Message] = []
        self._token_count: int = 0
        self._append_lock = asyncio.Lock()

        if isinstance(compactor, AsyncCompactor):
            self._lg.warning(
                "AsyncCompactor detected - use append_async() to avoid blocking",
                extra={"compactor": type(compactor).__name__},
            )

    # --- ConversationLike protocol ---

    def append(self, msg: Message) -> None:
        """Append a message (ConversationLike protocol).

        Raises ContextOverflowError BEFORE adding if the message cannot fit.

        Raises:
            RuntimeError: If AsyncCompactor configured. Use append_async().
            ContextOverflowError: If the message cannot fit within max_tokens.
        """
        tokens = estimate_message_tokens(
            msg.role, msg.content, msg.tool_calls, tokenizer=self.config.tokenizer
        )
        projected = self._token_count + tokens

        self._check_async_compactor_sync(projected)
        compacted = self._maybe_compact_sync(projected)
        if compacted:
            projected = self._token_count + tokens

        self._ensure_fits(tokens, projected)
        self._messages.append(msg)
        self._token_count += tokens

        if not compacted and self.compactor is not None and self.needs_compaction():
            self._run_compaction()

    async def append_async(self, msg: Message) -> None:
        """Append a message asynchronously (AsyncConversationLike protocol).

        Raises ContextOverflowError BEFORE adding if the message cannot fit.

        Raises:
            ContextOverflowError: If the message cannot fit within max_tokens.
        """
        async with self._append_lock:
            tokens = estimate_message_tokens(
                msg.role, msg.content, msg.tool_calls, tokenizer=self.config.tokenizer
            )
            projected = self._token_count + tokens

            compacted = await self._maybe_compact_async(projected)
            if compacted:
                projected = self._token_count + tokens

            self._ensure_fits(tokens, projected)
            self._messages.append(msg)
            self._token_count += tokens

            if not compacted and self.compactor is not None and self.needs_compaction():
                await self._run_compaction_async()

    def _should_compact(self, projected: int) -> bool:
        """Check if compaction should be triggered for the projected token count."""
        if self.compactor is None or self.config.max_tokens <= 0:
            return False
        would_exceed = projected > self.config.max_tokens
        would_trigger = projected / self.config.max_tokens >= self.config.compact_threshold
        return would_exceed or would_trigger

    def _check_async_compactor_sync(self, projected: int) -> None:
        """Raise RuntimeError if async compactor would be triggered in sync context."""
        if isinstance(self.compactor, AsyncCompactor) and self._should_compact(projected):
            raise RuntimeError(
                "Async compactor requires append_async(). "
                "Use 'await conversation.append_async(msg)' instead."
            )

    def _maybe_compact_sync(self, projected: int) -> bool:
        """Compact if needed (sync). Returns True if compaction ran."""
        if self._should_compact(projected):
            self._run_compaction()
            return True
        return False

    async def _maybe_compact_async(self, projected: int) -> bool:
        """Compact if needed (async). Returns True if compaction ran."""
        if self._should_compact(projected):
            await self._run_compaction_async()
            return True
        return False

    def _ensure_fits(self, msg_tokens: int, projected: int) -> None:
        """Raise ContextOverflowError if projected tokens exceed max_tokens."""
        if self.config.max_tokens > 0 and projected > self.config.max_tokens:
            self._log_context_overflow(msg_tokens, projected)
            raise ContextOverflowError(
                f"Message ({msg_tokens} tokens) would exceed max_tokens={self.config.max_tokens}. "
                f"Current: {self._token_count}, projected: {projected}.",
                token_count=projected,
                max_tokens=self.config.max_tokens,
            )

    def _run_compaction(self) -> None:
        """Run compaction and log stats."""
        assert self.compactor is not None  # noqa: S101
        before_msgs, before_tokens, t0 = self._log_compaction_start()
        self.compactor.compact(self)
        self._log_compaction_end(before_msgs, before_tokens, t0)

    async def _run_compaction_async(self) -> None:
        """Run compaction asynchronously and log stats."""
        assert self.compactor is not None  # noqa: S101
        before_msgs, before_tokens, t0 = self._log_compaction_start()

        if isinstance(self.compactor, AsyncCompactor):
            await self.compactor.compact(self)
        else:
            await asyncio.to_thread(self.compactor.compact, self)

        self._log_compaction_end(before_msgs, before_tokens, t0)

    def _log_compaction_start(self) -> tuple[int, int, float]:
        """Log compaction start and return state for later comparison."""
        before_msgs = len(self._messages)
        before_tokens = self._token_count
        self._lg.trace(
            "compacting conversation...",
            extra={"messages": before_msgs, "tokens": before_tokens},
        )
        return before_msgs, before_tokens, start()

    def _log_compaction_end(self, before_msgs: int, before_tokens: int, t0: float) -> None:
        """Log compaction completion with stats."""
        assert self.compactor is not None  # noqa: S101
        after_msgs = len(self._messages)
        after_tokens = self._token_count
        self._lg.debug(
            "conversation compacted",
            extra={
                "after": since(t0),
                "compactor": type(self.compactor).__name__,
                "messages": {"before": before_msgs, "after": after_msgs},
                "tokens": {
                    "before": before_tokens,
                    "after": after_tokens,
                    "saved": before_tokens - after_tokens,
                },
                "usage_ratio": self.usage_ratio,
            },
        )

    def _log_context_overflow(self, msg_tokens: int, projected: int) -> None:
        """Log context overflow before raising error."""
        self._lg.error(
            "context overflow: message cannot fit within max_tokens",
            extra={
                "message_tokens": msg_tokens,
                "current_tokens": self._token_count,
                "projected_tokens": projected,
                "max_tokens": self.config.max_tokens,
                "message_count": len(self._messages),
            },
        )

    def as_messages(self) -> list[Message]:
        """Return current messages as a view (ConversationLike protocol).

        Returns the internal list directly — saia calls this fresh before each
        LLM call, and compaction mutations between iterations are visible.
        """
        return self._messages

    # --- Convenience methods ---

    def add(
        self,
        content: str,
        role: str | Role = Role.USER,
        *,
        tool_calls: list[ToolCall] | None = None,
        tool_call_id: str | None = None,
    ) -> None:
        """Add a message to the conversation.

        Convenience wrapper around ``append()`` that constructs a ``Message``.

        Args:
            content: Message text content.
            role: Message role (default: Role.USER).
            tool_calls: Tool calls requested by assistant (assistant messages only).
            tool_call_id: ID of the tool call this responds to (tool messages only).
        """
        self.append(
            Message(
                role=role,
                content=content,
                tool_calls=tool_calls,
                tool_call_id=tool_call_id,
            )
        )

    @property
    def messages(self) -> list[Message]:
        """Get all messages (copy).

        Returns:
            Copy of message list. Use ``as_messages()`` for the live view.
        """
        return list(self._messages)

    def messages_as_dicts(self) -> list[dict[str, Any]]:
        """Get messages as plain dicts, omitting None-valued fields.

        Suitable for passing directly to LLM client APIs that reject
        unexpected null fields (e.g. ``tool_calls: null`` on user messages).
        """
        return [
            {k: v for k, v in dataclasses.asdict(m).items() if v is not None}
            for m in self._messages
        ]

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
            estimate_message_tokens(
                m.role, m.content, m.tool_calls, tokenizer=self.config.tokenizer
            )
            for m in self._messages
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

        Preserves system messages (if ``preserve_system``) and at least
        ``min_recent_messages`` recent messages. The cut is walked backward
        to never bisect an assistant/tool group, since the chat protocol
        requires every tool_call_id to have a matching assistant.tool_calls
        in the same request. This may preserve more than ``min_recent_messages``.

        Returns:
            Tuple of (messages_to_compact, messages_to_preserve).
        """
        preserve_count = max(0, self.config.min_recent_messages)

        if self.config.preserve_system:
            system_msgs = [m for m in self._messages if m.role == Role.SYSTEM]
            pool = [m for m in self._messages if m.role != Role.SYSTEM]
        else:
            system_msgs = []
            pool = list(self._messages)

        if len(pool) <= preserve_count:
            return [], list(self._messages)

        if preserve_count == 0:
            return list(pool), list(system_msgs)

        cut = _adjust_cut_to_tool_group_boundary(pool, len(pool) - preserve_count)
        if cut == 0:
            return [], list(self._messages)
        return pool[:cut], system_msgs + pool[cut:]

    def __len__(self) -> int:
        """Number of messages."""
        return len(self._messages)

    # --- Serialization ---

    def to_dict(self) -> dict[str, Any]:
        """Serialize conversation state for persistence.

        Returns dict with messages and token count. Config and compactor are not
        serialized — pass them fresh when restoring via from_dict(). Token count
        is included for informational purposes but is recalculated on restore.
        """
        return {
            "messages": [m.to_dict() for m in self._messages],
            "token_count": self._token_count,
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        lg: Logger,
        config: Config | None = None,
        compactor: Compactor | AsyncCompactor | None = None,
    ) -> Self:
        """Restore conversation from serialized state.

        Token count is recalculated using the provided config's tokenizer,
        ensuring consistency regardless of the tokenizer used at serialization.

        Args:
            data: Dict from to_dict() containing messages.
            lg: Logger instance.
            config: Conversation config (uses defaults if None).
            compactor: Optional compactor for automatic compaction.

        Returns:
            Restored Conversation instance.

        Raises:
            ValueError: If data is malformed (missing 'messages' key or not a list).
        """
        if not isinstance(data, dict):
            raise ValueError(f"Expected dict, got {type(data).__name__}")
        if "messages" not in data:
            raise ValueError("Missing required key 'messages'")
        if not isinstance(data["messages"], list):
            raise ValueError(
                f"Expected 'messages' to be a list, got {type(data['messages']).__name__}"
            )
        for i, m in enumerate(data["messages"]):
            if not isinstance(m, dict):
                raise ValueError(f"Expected messages[{i}] to be a dict, got {type(m).__name__}")

        conv = cls(lg, config=config, compactor=compactor)
        conv.replace_messages([Message.from_dict(m) for m in data["messages"]])
        return conv
