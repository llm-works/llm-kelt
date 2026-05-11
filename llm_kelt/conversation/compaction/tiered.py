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

from ..types import Message, Role
from .base import AsyncCompactor, Compactor

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


class TieredCompactor(Compactor):
    """Two-phase compaction: trim tool outputs first, summarize later.

    Phase 1: Truncate large tool results (web searches, fetched pages)
    while preserving the tool call structure. This removes bulk data
    but keeps the conversation flow intact.

    Phase 2: If still over threshold after trimming, summarize older
    messages using an LLM call.

    Note: This compactor makes a synchronous LLM call in phase 2.
    Use ``AsyncTieredCompactor`` for non-blocking summarization.

    Args:
        client: LLM client for summarization (phase 2).
        model: Model to use for summarization (optional).
        trim_threshold: Character count above which tool results get trimmed.
        trimmable_tools: Set of tool names whose output can be trimmed.
            Defaults to common bulk-data tools (web_search, web_fetch, etc.).
        summary_prompt: Custom prompt for phase 2 summarization.
    """

    def __init__(
        self,
        client: ChatClient,
        model: str | None = None,
        trim_threshold: int = _DEFAULT_TRIM_THRESHOLD,
        trimmable_tools: frozenset[str] | set[str] | None = None,
        summary_prompt: str | None = None,
    ) -> None:
        self._client = client
        self._model = model
        self._trim_threshold = trim_threshold
        self._trimmable_tools = (
            frozenset(trimmable_tools) if trimmable_tools else DEFAULT_TRIMMABLE_TOOLS
        )
        self._summary_prompt = summary_prompt or _SUMMARY_PROMPT

    def compact(self, conversation: Conversation) -> None:
        """Compact conversation using tiered strategy."""
        # Phase 1: Trim tool results
        trimmed = _trim_tool_results(conversation, self._trim_threshold, self._trimmable_tools)
        if trimmed and not conversation.needs_compaction():
            return  # Trimming was sufficient

        # Phase 2: Summarize if still over threshold
        self._summarize(conversation)

    def _summarize(self, conversation: Conversation) -> None:
        """Phase 2: Summarize older messages using LLM."""
        to_compact, preserved = conversation.split_for_compaction()
        if not to_compact:
            return

        summary = self._generate_summary(to_compact)
        new_messages = _build_compacted_messages(preserved, summary)
        conversation.replace_messages(new_messages)

    def _generate_summary(self, messages: list[Message]) -> str:
        """Generate summary of messages using LLM."""
        formatted = _format_messages(messages)

        kwargs: dict = {"temperature": 0.3}
        if self._model:
            kwargs["model"] = self._model

        try:
            response = self._client.chat(
                messages=[
                    {"role": "system", "content": self._summary_prompt},
                    {"role": "user", "content": formatted},
                ],
                **kwargs,
            )
            return response.content or "[Summary unavailable]"
        except Exception:
            return "[Summarization failed - continuing with trimmed context]"


class AsyncTieredCompactor(AsyncCompactor):
    """Async version of TieredCompactor for non-blocking summarization.

    Same two-phase strategy as ``TieredCompactor``, but uses async LLM
    calls in phase 2 to avoid blocking the event loop.

    Args:
        client: LLM client for summarization (phase 2).
        model: Model to use for summarization (optional).
        trim_threshold: Character count above which tool results get trimmed.
        trimmable_tools: Set of tool names whose output can be trimmed.
        summary_prompt: Custom prompt for phase 2 summarization.
    """

    def __init__(
        self,
        client: ChatClient,
        model: str | None = None,
        trim_threshold: int = _DEFAULT_TRIM_THRESHOLD,
        trimmable_tools: frozenset[str] | set[str] | None = None,
        summary_prompt: str | None = None,
    ) -> None:
        self._client = client
        self._model = model
        self._trim_threshold = trim_threshold
        self._trimmable_tools = (
            frozenset(trimmable_tools) if trimmable_tools else DEFAULT_TRIMMABLE_TOOLS
        )
        self._summary_prompt = summary_prompt or _SUMMARY_PROMPT

    async def compact(self, conversation: Conversation) -> None:
        """Compact conversation using tiered strategy."""
        # Phase 1: Trim tool results (sync - no I/O)
        trimmed = _trim_tool_results(conversation, self._trim_threshold, self._trimmable_tools)
        if trimmed and not conversation.needs_compaction():
            return  # Trimming was sufficient

        # Phase 2: Summarize if still over threshold
        await self._summarize(conversation)

    async def _summarize(self, conversation: Conversation) -> None:
        """Phase 2: Summarize older messages using LLM."""
        to_compact, preserved = conversation.split_for_compaction()
        if not to_compact:
            return

        summary = await self._generate_summary(to_compact)
        new_messages = _build_compacted_messages(preserved, summary)
        conversation.replace_messages(new_messages)

    async def _generate_summary(self, messages: list[Message]) -> str:
        """Generate summary of messages using async LLM call."""
        formatted = _format_messages(messages)

        kwargs: dict = {"temperature": 0.3}
        if self._model:
            kwargs["model"] = self._model

        try:
            response = await self._client.chat_async(
                messages=[
                    {"role": "system", "content": self._summary_prompt},
                    {"role": "user", "content": formatted},
                ],
                **kwargs,
            )
            return response.content or "[Summary unavailable]"
        except Exception:
            return "[Summarization failed - continuing with trimmed context]"


# --- Shared helpers ---


def _trim_tool_results(
    conversation: Conversation,
    trim_threshold: int,
    trimmable_tools: frozenset[str],
) -> bool:
    """Trim large tool results, preserving structure. Returns True if any trimming done."""
    messages = conversation.as_messages()
    tool_names = _build_tool_name_map(messages)

    modified = False
    new_messages: list[Message] = []

    for msg in messages:
        if msg.role == Role.TOOL and _should_trim(msg, tool_names, trim_threshold, trimmable_tools):
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
) -> bool:
    """Check if a tool result should be trimmed."""
    if not msg.tool_call_id:
        return False

    tool_name = tool_names.get(msg.tool_call_id, "")
    is_trimmable = tool_name in trimmable_tools
    content_len = len(msg.content or "")
    is_large = content_len > trim_threshold

    return is_trimmable and is_large


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


def _build_compacted_messages(preserved: list[Message], summary: str) -> list[Message]:
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
    """Format messages for summarization."""
    lines = []
    for msg in messages:
        if msg.role == Role.TOOL:
            content = (msg.content or "")[:500]
            lines.append(f"TOOL RESULT: {content}")
        elif msg.tool_calls:
            tools = ", ".join(tc.name for tc in msg.tool_calls)
            lines.append(f"ASSISTANT [called: {tools}]: {msg.content or ''}")
        else:
            role = msg.role.upper() if isinstance(msg.role, str) else msg.role.name
            lines.append(f"{role}: {msg.content or ''}")
    return "\n\n".join(lines)
