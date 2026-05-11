"""Shared helpers for compaction strategies."""

from __future__ import annotations

from ..types import Message, Role


def build_compacted_messages(preserved: list[Message], summary: str) -> list[Message]:
    """Build the final message list with summary inserted after system messages."""
    new_messages: list[Message] = []

    system_msgs = [m for m in preserved if m.role == Role.SYSTEM]
    non_system = [m for m in preserved if m.role != Role.SYSTEM]
    new_messages.extend(system_msgs)

    new_messages.append(
        Message(
            role=Role.USER,
            content=f"[Previous conversation summary]\n{summary}\n[End summary]",
        )
    )

    new_messages.extend(non_system)
    return new_messages


def format_messages_for_summary(
    messages: list[Message], truncate_tool_content: int | None = None
) -> str:
    """Format messages as text for summarization.

    Args:
        messages: Messages to format.
        truncate_tool_content: If set, truncate tool result content to this many chars.
    """
    lines = []
    for msg in messages:
        if msg.role == Role.TOOL:
            content = msg.content or ""
            if truncate_tool_content is not None:
                content = content[:truncate_tool_content]
            lines.append(f"TOOL RESULT: {content}")
        elif msg.tool_calls:
            tools = ", ".join(tc.name for tc in msg.tool_calls)
            lines.append(f"ASSISTANT [called: {tools}]: {msg.content or ''}")
        else:
            role = msg.role.upper() if isinstance(msg.role, str) else msg.role.name
            lines.append(f"{role}: {msg.content or ''}")
    return "\n\n".join(lines)
