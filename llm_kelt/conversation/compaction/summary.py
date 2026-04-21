"""Summarizing compaction strategy.

Uses an LLM to summarize older messages before discarding them.
More expensive (requires LLM call) but retains context better
than simple window-based compaction.

Supports guards for quality assurance with retry/escalation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..tokens import Tokenizer, estimate_tokens
from ..types import Message, Role
from .base import AsyncCompactor
from .guard import CompactionContext, CompactionGuard, CompactionGuardError

if TYPE_CHECKING:
    from llm_infer.client import ChatClient

    from ..session import Conversation

_DEFAULT_SUMMARY_PROMPT = (
    "Summarize the following conversation concisely, preserving key information, "
    "decisions made, and important context. Focus on facts and outcomes, not "
    "conversational filler."
)


class SummarizingCompactor(AsyncCompactor):
    """Compactor that summarizes older messages using an LLM.

    Preserves information by creating a summary of compacted messages,
    which is inserted as a user context message after the system message.

    Supports guards for validating summary quality with retry/escalation.

    Args:
        client: LLM client for generating summaries (uses async API).
        model: Model to use (optional, uses client default).
        summary_prompt: Custom prompt for summarization.
        guards: Optional guards for validating compaction quality.
        tokenizer: Optional tokenizer for accurate token counting in guards.
    """

    def __init__(
        self,
        client: ChatClient,
        model: str | None = None,
        summary_prompt: str | None = None,
        guards: list[CompactionGuard] | None = None,
        tokenizer: Tokenizer | None = None,
    ) -> None:
        self._client = client
        self._model = model
        self._summary_prompt = summary_prompt or _DEFAULT_SUMMARY_PROMPT
        self._guards = guards or []
        self._tokenizer = tokenizer

    async def compact(self, conversation: Conversation) -> None:
        """Summarize old messages and replace with summary."""
        to_compact, preserved = conversation.split_for_compaction()
        if not to_compact:
            return

        before_tokens = conversation.token_count
        before_messages = list(conversation.messages)

        summary = await self._summarize_with_guards(to_compact, before_messages, before_tokens)

        new_messages = self._build_compacted_messages(preserved, summary)
        conversation.replace_messages(new_messages)

    async def _summarize_with_guards(
        self,
        to_compact: list[Message],
        before_messages: list[Message],
        before_tokens: int,
    ) -> str:
        """Generate summary with guard validation and retry logic."""
        formatted = _format_messages(to_compact)
        max_attempts = 1 + sum(g.max_retries for g in self._guards)
        last_error: str | None = None
        last_feedback: str | None = None

        for attempt in range(max_attempts):
            prompt = self._build_prompt(formatted, last_error, last_feedback, attempt)
            summary = await self._call_llm(prompt)

            if not self._guards:
                return summary

            ctx = self._build_context(before_messages, summary, before_tokens, attempt)
            error, feedback = self._run_guards(ctx)

            if error is None:
                return summary

            last_error = error
            last_feedback = feedback

        raise CompactionGuardError(
            guard_name=None,
            error=last_error or "Unknown error",
            attempts=max_attempts,
        )

    def _build_prompt(
        self,
        formatted: str,
        last_error: str | None,
        last_feedback: str | None,
        attempt: int,
    ) -> str:
        """Build summarization prompt, including retry feedback if applicable."""
        if attempt == 0 or last_feedback is None:
            return formatted
        return f"{formatted}\n\n---\n\n{last_feedback}"

    async def _call_llm(self, content: str) -> str:
        """Call the LLM to generate a summary."""
        kwargs: dict = {"temperature": 0.3}
        if self._model:
            kwargs["model"] = self._model

        response = await self._client.chat_async(
            messages=[
                {"role": "system", "content": self._summary_prompt},
                {"role": "user", "content": content},
            ],
            **kwargs,
        )
        return response.content or "[Summary unavailable]"

    def _build_context(
        self,
        before_messages: list[Message],
        summary: str,
        before_tokens: int,
        attempt: int,
    ) -> CompactionContext:
        """Build context for guard validation."""
        summary_tokens = estimate_tokens(summary, tokenizer=self._tokenizer)
        # Estimate after_tokens as summary + preserved overhead
        after_tokens = summary_tokens + 50  # rough estimate for structure overhead

        return CompactionContext(
            before_messages=before_messages,
            after_messages=[],  # Not yet constructed
            before_tokens=before_tokens,
            after_tokens=after_tokens,
            summary=summary,
            attempt=attempt,
        )

    def _run_guards(self, ctx: CompactionContext) -> tuple[str | None, str | None]:
        """Run all guards, return (first_error, combined_feedback) or (None, None)."""
        errors: list[str] = []
        feedbacks: list[str] = []

        for guard in self._guards:
            error = guard.validator(ctx)
            if error is not None:
                errors.append(error)
                feedback = guard.resolve_instruction(ctx.attempt, ctx, error)
                feedbacks.append(feedback)

        if not errors:
            return None, None

        return errors[0], "\n\n".join(feedbacks)

    def _build_compacted_messages(self, preserved: list[Message], summary: str) -> list[Message]:
        """Build the final message list with summary inserted."""
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
