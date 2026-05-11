"""Tiered compaction strategy.

Implements a two-phase compaction approach that preserves conversation
structure better than all-or-nothing summarization:

1. First trim large tool outputs (web fetches, search results, etc.)
2. Only summarize if still over threshold

This keeps agent reasoning and decisions intact longer since bulk data
(fetched pages, search results) gets trimmed first - it's least valuable
to retain verbatim.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..tokens import Tokenizer, estimate_message_tokens, estimate_tokens
from ..types import Message, Role
from ._helpers import build_compacted_messages, format_messages_for_summary
from .base import AsyncCompactor, Compactor
from .guard import CompactionContext, CompactionGuard, CompactionGuardError

if TYPE_CHECKING:
    from llm_infer.client import ChatClient

    from ..session import Conversation

# Tools whose output is safe to aggressively trim (bulk data, not decisions)
DEFAULT_TRIMMABLE_TOOLS: frozenset[str] = frozenset(
    {
        "web_search",
        "web_fetch",
        "search_tweets",
        "get_user_tweets",
        "read_file",
        "grep",
    }
)

_DEFAULT_TRIM_THRESHOLD = 2000

_SUMMARY_PROMPT = (
    "Summarize this conversation VERY concisely. Target: 50% reduction "
    "or more from the input length.\n\n"
    "MUST preserve (briefly):\n"
    "- Key findings and decisions made\n"
    "- Important entities and their relationships\n"
    "- Critical context needed for continuation\n"
    "- Source URLs if present\n\n"
    "AGGRESSIVELY omit:\n"
    "- Raw search results, fetched page content\n"
    "- Routine tool outputs and intermediate steps\n"
    "- Redundant information already captured\n"
    "- Detailed reasoning that can be inferred from actions\n\n"
    "NEVER fabricate URLs or citations. Only include URLs from the input.\n\n"
    "Be terse. Use bullet points. No preamble."
)


class _TieredMixin:
    """Shared configuration and helpers for tiered compactors."""

    _client: ChatClient
    _model: str | None
    _trim_threshold: int
    _trim_threshold_tokens: int | None
    _trimmable_tools: frozenset[str]
    _summary_prompt: str
    _guards: list[CompactionGuard]
    _tokenizer: Tokenizer | None

    def _init_tiered(
        self,
        client: ChatClient,
        model: str | None = None,
        trim_threshold: int = _DEFAULT_TRIM_THRESHOLD,
        trim_threshold_tokens: int | None = None,
        trimmable_tools: frozenset[str] | set[str] | None = None,
        summary_prompt: str | None = None,
        guards: list[CompactionGuard] | None = None,
        tokenizer: Tokenizer | None = None,
    ) -> None:
        """Initialize tiered compactor config."""
        self._client = client
        self._model = model
        self._trim_threshold = trim_threshold
        self._trim_threshold_tokens = trim_threshold_tokens
        self._trimmable_tools = (
            frozenset(trimmable_tools) if trimmable_tools else DEFAULT_TRIMMABLE_TOOLS
        )
        self._summary_prompt = summary_prompt or _SUMMARY_PROMPT
        self._guards = guards or []
        self._tokenizer = tokenizer

    def _run_phase1(self, conversation: Conversation) -> bool:
        """Phase 1: Trim tool results. Returns True if trimming was sufficient."""
        tokenizer = None
        if self._trim_threshold_tokens is not None:
            tokenizer = self._tokenizer or conversation.config.tokenizer
        trimmed = _trim_tool_results(
            conversation,
            self._trim_threshold,
            self._trimmable_tools,
            trim_threshold_tokens=self._trim_threshold_tokens,
            tokenizer=tokenizer,
        )
        return trimmed and not conversation.needs_compaction()

    def _prepare_summary(
        self, conversation: Conversation
    ) -> tuple[list[Message], list[Message], list[Message], int, Tokenizer | None] | None:
        """Prepare for phase 2. Returns context needed for summarization or None."""
        to_compact, preserved = conversation.split_for_compaction()
        if not to_compact:
            return None
        before_messages = list(conversation.messages)
        tokenizer = self._tokenizer or conversation.config.tokenizer
        before_tokens = sum(
            estimate_message_tokens(m.role, m.content, m.tool_calls, tokenizer=tokenizer)
            for m in before_messages
        )
        return to_compact, preserved, before_messages, before_tokens, tokenizer

    def _finalize_summary(
        self, conversation: Conversation, preserved: list[Message], summary: str
    ) -> None:
        """Apply the summary to the conversation."""
        new_messages = build_compacted_messages(preserved, summary)
        conversation.replace_messages(new_messages)

    def _build_summary_request(
        self, messages: list[Message], feedback: str | None = None, attempt: int = 0
    ) -> tuple[list[dict], dict]:
        """Build LLM request for summary generation."""
        formatted = format_messages_for_summary(messages, truncate_tool_content=500)
        if attempt > 0 and feedback:
            formatted = f"{formatted}\n\n---\n\n{feedback}"
        llm_messages = [
            {"role": "system", "content": self._summary_prompt},
            {"role": "user", "content": formatted},
        ]
        kwargs: dict = {"temperature": 0.3}
        if self._model:
            kwargs["model"] = self._model
        return llm_messages, kwargs

    def _build_guard_context(
        self,
        preserved: list[Message],
        before_messages: list[Message],
        summary: str,
        before_tokens: int,
        attempt: int,
        tokenizer: Tokenizer | None,
    ) -> CompactionContext:
        """Build context for guard validation."""
        after_messages = build_compacted_messages(preserved, summary)
        after_tokens = sum(
            estimate_message_tokens(m.role, m.content, m.tool_calls, tokenizer=tokenizer)
            for m in after_messages
        )
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

    def _check_guards(self, ctx: CompactionContext, guard_attempts: dict[int, int]) -> str | None:
        """Check guards and raise if any exhausted retries. Returns feedback or None."""
        for i, guard in enumerate(self._guards):
            if guard_attempts[i] > guard.max_retries:
                continue
            error = guard.validator(ctx)
            if error is not None:
                guard_attempts[i] += 1
                if guard_attempts[i] > guard.max_retries:
                    raise CompactionGuardError(
                        guard_name=guard.name,
                        error=error,
                        attempts=guard_attempts[i],
                    )
                return guard.resolve_instruction(guard_attempts[i] - 1, ctx, error)
        return None


class TieredCompactor(_TieredMixin, Compactor):
    """Two-phase compaction: trim tool outputs first, summarize later.

    Phase 1: Truncate large tool results (web searches, fetched pages)
    while preserving the tool call structure. This removes bulk data
    but keeps the conversation flow intact.

    Phase 2: If still over threshold after trimming, summarize older
    messages using an LLM call.

    Supports guards for validating summary quality with retry/escalation.

    Note: This compactor makes synchronous LLM calls in phase 2.
    Use ``AsyncTieredCompactor`` for non-blocking summarization.

    Args:
        client: LLM client for summarization (phase 2).
        model: Model to use for summarization (optional).
        trim_threshold: Character count above which tool results get trimmed.
        trim_threshold_tokens: Token count above which tool results get trimmed.
            When set, uses tokenizer for accurate counting instead of chars.
        trimmable_tools: Set of tool names whose output can be trimmed.
            Defaults to common bulk-data tools (web_search, web_fetch, etc.).
        summary_prompt: Custom prompt for phase 2 summarization.
        guards: Optional guards for validating compaction quality.
        tokenizer: Optional tokenizer for token counting.
    """

    def __init__(
        self,
        client: ChatClient,
        model: str | None = None,
        trim_threshold: int = _DEFAULT_TRIM_THRESHOLD,
        trim_threshold_tokens: int | None = None,
        trimmable_tools: frozenset[str] | set[str] | None = None,
        summary_prompt: str | None = None,
        guards: list[CompactionGuard] | None = None,
        tokenizer: Tokenizer | None = None,
    ) -> None:
        self._init_tiered(
            client,
            model,
            trim_threshold,
            trim_threshold_tokens,
            trimmable_tools,
            summary_prompt,
            guards,
            tokenizer,
        )

    def compact(self, conversation: Conversation) -> None:
        """Compact conversation using tiered strategy."""
        if self._run_phase1(conversation):
            return
        prepared = self._prepare_summary(conversation)
        if not prepared:
            return
        to_compact, preserved, before_messages, before_tokens, tokenizer = prepared
        summary = self._summarize_with_guards(
            to_compact, preserved, before_messages, before_tokens, tokenizer
        )
        self._finalize_summary(conversation, preserved, summary)

    def _summarize_with_guards(
        self,
        to_compact: list[Message],
        preserved: list[Message],
        before_messages: list[Message],
        before_tokens: int,
        tokenizer: Tokenizer | None,
    ) -> str:
        """Generate summary with guard validation and retry logic."""
        if not self._guards:
            return self._call_llm(to_compact)

        guard_attempts: dict[int, int] = {i: 0 for i in range(len(self._guards))}
        feedback: str | None = None
        attempt = 0

        while True:
            summary = self._call_llm(to_compact, feedback, attempt)
            ctx = self._build_guard_context(
                preserved, before_messages, summary, before_tokens, attempt, tokenizer
            )
            feedback = self._check_guards(ctx, guard_attempts)
            if feedback is None:
                return summary
            attempt += 1

    def _call_llm(
        self, messages: list[Message], feedback: str | None = None, attempt: int = 0
    ) -> str:
        """Call the LLM to generate a summary."""
        llm_messages, kwargs = self._build_summary_request(messages, feedback, attempt)
        response = self._client.chat(messages=llm_messages, **kwargs)
        return response.content or "[Summary unavailable]"


class AsyncTieredCompactor(_TieredMixin, AsyncCompactor):
    """Async version of TieredCompactor for non-blocking summarization.

    Same two-phase strategy as ``TieredCompactor``, but uses async LLM
    calls in phase 2 to avoid blocking the event loop.

    Supports guards for validating summary quality with retry/escalation.

    Args:
        client: LLM client for summarization (phase 2).
        model: Model to use for summarization (optional).
        trim_threshold: Character count above which tool results get trimmed.
        trim_threshold_tokens: Token count above which tool results get trimmed.
            When set, uses tokenizer for accurate counting instead of chars.
        trimmable_tools: Set of tool names whose output can be trimmed.
        summary_prompt: Custom prompt for phase 2 summarization.
        guards: Optional guards for validating compaction quality.
        tokenizer: Optional tokenizer for token counting.
    """

    def __init__(
        self,
        client: ChatClient,
        model: str | None = None,
        trim_threshold: int = _DEFAULT_TRIM_THRESHOLD,
        trim_threshold_tokens: int | None = None,
        trimmable_tools: frozenset[str] | set[str] | None = None,
        summary_prompt: str | None = None,
        guards: list[CompactionGuard] | None = None,
        tokenizer: Tokenizer | None = None,
    ) -> None:
        self._init_tiered(
            client,
            model,
            trim_threshold,
            trim_threshold_tokens,
            trimmable_tools,
            summary_prompt,
            guards,
            tokenizer,
        )

    async def compact(self, conversation: Conversation) -> None:
        """Compact conversation using tiered strategy."""
        if self._run_phase1(conversation):
            return
        prepared = self._prepare_summary(conversation)
        if not prepared:
            return
        to_compact, preserved, before_messages, before_tokens, tokenizer = prepared
        summary = await self._summarize_with_guards(
            to_compact, preserved, before_messages, before_tokens, tokenizer
        )
        self._finalize_summary(conversation, preserved, summary)

    async def _summarize_with_guards(
        self,
        to_compact: list[Message],
        preserved: list[Message],
        before_messages: list[Message],
        before_tokens: int,
        tokenizer: Tokenizer | None,
    ) -> str:
        """Generate summary with guard validation and retry logic."""
        if not self._guards:
            return await self._call_llm(to_compact)

        guard_attempts: dict[int, int] = {i: 0 for i in range(len(self._guards))}
        feedback: str | None = None
        attempt = 0

        while True:
            summary = await self._call_llm(to_compact, feedback, attempt)
            ctx = self._build_guard_context(
                preserved, before_messages, summary, before_tokens, attempt, tokenizer
            )
            feedback = self._check_guards(ctx, guard_attempts)
            if feedback is None:
                return summary
            attempt += 1

    async def _call_llm(
        self, messages: list[Message], feedback: str | None = None, attempt: int = 0
    ) -> str:
        """Call the LLM to generate a summary."""
        llm_messages, kwargs = self._build_summary_request(messages, feedback, attempt)
        response = await self._client.chat_async(messages=llm_messages, **kwargs)
        return response.content or "[Summary unavailable]"


# --- Shared helpers ---


def _trim_tool_results(
    conversation: Conversation,
    trim_threshold: int,
    trimmable_tools: frozenset[str],
    trim_threshold_tokens: int | None = None,
    tokenizer: Tokenizer | None = None,
) -> bool:
    """Trim large tool results, preserving structure. Returns True if any trimming done."""
    messages = conversation.as_messages()
    tool_names = _build_tool_name_map(messages)

    modified = False
    new_messages: list[Message] = []

    for msg in messages:
        should_trim = msg.role == Role.TOOL and _should_trim(
            msg, tool_names, trim_threshold, trimmable_tools, trim_threshold_tokens, tokenizer
        )
        if should_trim:
            trimmed_msg = _trim_message(msg, tool_names)
            new_messages.append(trimmed_msg)
            modified = True
        else:
            new_messages.append(msg)

    if modified:
        conversation.replace_messages(new_messages)

    return modified


def _build_tool_name_map(messages: list[Message]) -> dict[str, str]:
    """Map tool_call_id -> tool_name from assistant messages."""
    tool_names: dict[str, str] = {}
    for msg in messages:
        if msg.tool_calls:
            for tc in msg.tool_calls:
                tool_names[tc.id] = tc.name
    return tool_names


def _should_trim(
    msg: Message,
    tool_names: dict[str, str],
    trim_threshold: int,
    trimmable_tools: frozenset[str],
    trim_threshold_tokens: int | None = None,
    tokenizer: Tokenizer | None = None,
) -> bool:
    """Check if a tool result should be trimmed."""
    if not msg.tool_call_id:
        return False

    tool_name = tool_names.get(msg.tool_call_id, "")
    if tool_name not in trimmable_tools:
        return False

    content = msg.content or ""
    if trim_threshold_tokens is not None and tokenizer is not None:
        content_size = estimate_tokens(content, tokenizer=tokenizer)
        return content_size > trim_threshold_tokens

    return len(content) > trim_threshold


def _trim_message(msg: Message, tool_names: dict[str, str]) -> Message:
    """Create a trimmed version of a tool result message."""
    tool_name = tool_names.get(msg.tool_call_id or "", "tool")
    content = msg.content or ""
    original_len = len(content)

    summary = _extract_summary(tool_name, content)

    return Message(
        role=msg.role,
        content=f"[{tool_name} result trimmed: {original_len} chars]\n{summary}",
        tool_call_id=msg.tool_call_id,
    )


def _extract_summary(tool_name: str, content: str) -> str:
    """Extract a useful summary from tool output."""
    if tool_name == "web_search":
        return _summarize_search_results(content)
    elif tool_name == "web_fetch":
        return _summarize_fetch_result(content)
    else:
        # Generic: first 500 chars + last 200 chars
        if len(content) > 800:
            return f"{content[:500]}\n...[trimmed]...\n{content[-200:]}"
        return content[:700]


def _summarize_search_results(content: str) -> str:
    """Extract titles/URLs from search results."""
    lines = content.split("\n")
    summary_lines = []
    for line in lines[:20]:
        line = line.strip()
        if line and (line.startswith("http") or "://" in line or len(line) < 200):
            summary_lines.append(line)
        if len(summary_lines) >= 10:
            break
    return "\n".join(summary_lines) if summary_lines else content[:500]


def _summarize_fetch_result(content: str) -> str:
    """Extract title and key content from fetched page."""
    lines = content.split("\n")
    summary_parts = []
    for line in lines[:10]:
        line = line.strip()
        if line and len(line) < 200:
            summary_parts.append(line)
    if summary_parts:
        return "\n".join(summary_parts[:5])
    return content[:500]
