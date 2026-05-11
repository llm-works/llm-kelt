"""Summarizing compaction strategy.

Uses an LLM to summarize older messages before discarding them.
More expensive (requires LLM call) but retains context better
than simple window-based compaction.

Supports guards for quality assurance with retry/escalation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..tokens import Tokenizer, estimate_message_tokens, estimate_tokens
from ..types import Message
from ._helpers import build_compacted_messages, format_messages_for_summary
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
        tokenizer: Optional tokenizer for token counting in guards. If not provided,
            uses the conversation's Config.tokenizer (ensuring consistent counting).
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

        before_messages = list(conversation.messages)

        # Use compactor's tokenizer if set, otherwise use conversation's
        tokenizer = self._tokenizer or conversation.config.tokenizer

        # Calculate before_tokens using the resolved tokenizer for consistency
        before_tokens = sum(
            estimate_message_tokens(m.role, m.content, m.tool_calls, tokenizer=tokenizer)
            for m in before_messages
        )

        summary = await self._summarize_with_guards(
            to_compact, preserved, before_messages, before_tokens, tokenizer
        )

        new_messages = build_compacted_messages(preserved, summary)
        conversation.replace_messages(new_messages)

    async def _summarize_with_guards(
        self,
        to_compact: list[Message],
        preserved: list[Message],
        before_messages: list[Message],
        before_tokens: int,
        tokenizer: Tokenizer | None,
    ) -> str:
        """Generate summary with guard validation and retry logic."""
        formatted = format_messages_for_summary(to_compact)
        if not self._guards:
            return await self._call_llm(formatted)

        guard_attempts: dict[int, int] = {i: 0 for i in range(len(self._guards))}
        last_feedback: str | None = None
        attempt = 0

        while True:
            prompt = self._build_prompt(formatted, last_feedback, attempt)
            summary = await self._call_llm(prompt)

            ctx = self._build_context(
                preserved, before_messages, summary, before_tokens, attempt, tokenizer
            )
            result = self._check_guards(ctx, guard_attempts)
            if result is None:
                return summary

            last_feedback = result
            attempt += 1

    def _check_guards(self, ctx: CompactionContext, guard_attempts: dict[int, int]) -> str | None:
        """Check guards and raise if any exhausted retries. Returns feedback or None."""
        failed, error, feedback = self._run_guards_with_tracking(ctx, guard_attempts)
        if failed is None:
            return None

        guard_idx, guard = failed
        if guard_attempts[guard_idx] > guard.max_retries:
            raise CompactionGuardError(
                guard_name=guard.name,
                error=error or "Unknown error",
                attempts=guard_attempts[guard_idx],
            )
        return feedback

    def _build_prompt(self, formatted: str, last_feedback: str | None, attempt: int) -> str:
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
        preserved: list[Message],
        before_messages: list[Message],
        summary: str,
        before_tokens: int,
        attempt: int,
        tokenizer: Tokenizer | None,
    ) -> CompactionContext:
        """Build context for guard validation."""
        # Build the actual after_messages for accurate token counting
        after_messages = build_compacted_messages(preserved, summary)

        # Calculate actual after_tokens from the compacted messages
        after_tokens = sum(
            estimate_message_tokens(m.role, m.content, m.tool_calls, tokenizer=tokenizer)
            for m in after_messages
        )

        # Calculate summary-only tokens for max_summary_tokens guard
        summary_tokens = estimate_tokens(summary, tokenizer=tokenizer)

        return CompactionContext(
            before_messages=before_messages,
            after_messages=after_messages,
            before_tokens=before_tokens,
            after_tokens=after_tokens,
            summary=summary,
            summary_tokens=summary_tokens,
            attempt=attempt,
        )

    def _run_guards_with_tracking(
        self, ctx: CompactionContext, guard_attempts: dict[int, int]
    ) -> tuple[tuple[int, CompactionGuard] | None, str | None, str | None]:
        """Run guards and track attempts per guard.

        Returns:
            (failed_guard, error, feedback) where failed_guard is (index, guard) or None if passed.
        """
        for i, guard in enumerate(self._guards):
            # Skip guards that have exhausted retries
            if guard_attempts[i] > guard.max_retries:
                continue

            error = guard.validator(ctx)
            if error is not None:
                guard_attempts[i] += 1
                feedback = guard.resolve_instruction(guard_attempts[i] - 1, ctx, error)
                return (i, guard), error, feedback

        return None, None, None
