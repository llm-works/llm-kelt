"""Canonical message types for conversations.

These are the shared types used across the appstack ecosystem (kelt, saia, gent).
Using FieldDict provides dict-native behavior with typed fields — zero serialization
overhead for JSON storage, compatible with existing dict-based message APIs.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from appinfra import FieldDict, field


class Role(StrEnum):
    """Message roles in a conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ToolCall(FieldDict):
    """A tool invocation requested by the LLM.

    Attributes:
        id: Unique identifier for this tool call (used to match results).
        name: Tool function name.
        arguments: Arguments to pass to the tool.
    """

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


class Message(FieldDict):
    """A message in a conversation.

    Supports standard chat messages and tool-related messages:
    - system/user/assistant: Standard chat messages
    - assistant with tool_calls: LLM requesting tool execution
    - tool: Result of a tool execution

    Attributes:
        role: Message role ("system", "user", "assistant", "tool").
        content: Message text content.
        tool_calls: Tool calls requested by assistant (assistant messages only).
        tool_call_id: ID of the tool call this responds to (tool messages only).
    """

    role: str  # Role enum value or string — StrEnum serializes as string
    content: str
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
