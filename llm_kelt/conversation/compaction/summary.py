"""Summarizing compaction strategy.

Uses an LLM to summarize older messages before discarding them.
More expensive (requires LLM call) but retains context better
than simple window-based compaction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..types import Message, Role
from .base import Compactor

if TYPE_CHECKING:
    from llm_infer.client import ChatClient

    from ..session import Conversation

_DEFAULT_SUMMARY_PROMPT = (
    "Summarize the following conversation concisely, preserving key information, "
    "decisions made, and important context. Focus on facts and outcomes, not "
    "conversational filler."
)


class SummarizingCompactor(Compactor):
    """Compactor that summarizes older messages using an LLM.

    Preserves information by creating a summary of compacted messages,
    which is inserted as a user context message after the system message.

    Args:
        client: LLM client for generating summaries.
        model: Model to use (optional, uses client default).
        summary_prompt: Custom prompt for summarization.
    """

    def __init__(
        self,
        client: ChatClient,
        model: str | None = None,
        summary_prompt: str | None = None,
    ) -> None:
        self._client = client
        self._model = model
        self._summary_prompt = summary_prompt or _DEFAULT_SUMMARY_PROMPT

    def compact(self, conversation: Conversation) -> None:
        """Summarize old messages and replace with summary.

        Keeps system message and recent messages, summarizes the rest.
        """
        to_compact, preserved = conversation.split_for_compaction()
        if not to_compact:
            return

        summary = self._summarize(to_compact)

        new_messages: list[Message] = []

        # System messages first (if present)
        system_msgs = [m for m in preserved if m.role == Role.SYSTEM]
        non_system = [m for m in preserved if m.role != Role.SYSTEM]
        new_messages.extend(system_msgs)

        # Summary as context
        new_messages.append(
            Message(
                role=Role.USER,
                content=f"[Previous conversation summary]\n{summary}\n[End summary]",
            )
        )

        new_messages.extend(non_system)
        conversation.replace_messages(new_messages)

    def _summarize(self, messages: list[Message]) -> str:
        """Generate summary of messages using LLM."""
        formatted = _format_messages(messages)

        kwargs: dict = {"temperature": 0.3}
        if self._model:
            kwargs["model"] = self._model

        response = self._client.chat(
            messages=[
                {"role": "system", "content": self._summary_prompt},
                {"role": "user", "content": formatted},
            ],
            **kwargs,
        )
        return response.content or "[Summary unavailable]"


def _format_messages(messages: list[Message]) -> str:
    """Format messages as text for summarization."""
    lines = []
    for msg in messages:
        if msg.role == "tool":
            lines.append(f"TOOL RESULT: {msg.content}")
        elif msg.tool_calls:
            tools = ", ".join(tc.name for tc in msg.tool_calls)
            lines.append(f"ASSISTANT [called: {tools}]: {msg.content}")
        else:
            lines.append(f"{msg.role.upper()}: {msg.content}")
    return "\n\n".join(lines)
