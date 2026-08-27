# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-kelt Authors

"""Token estimation utilities for conversation management.

Provides fast, approximate token counting without requiring tokenizer dependencies.
Uses character-based heuristics calibrated for common LLM tokenizers.

For accurate counting, pass a tokenizer callable to Conversation config.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llm_saia import ToolCall

# Type alias for tokenizer functions: text -> token count
Tokenizer = Callable[[str], int]

# Average chars per token varies by model/tokenizer, but 4 is a reasonable default.
# GPT-style tokenizers: ~4 chars/token for English text.
#
# LIMITATION: This heuristic is calibrated for English/Latin scripts. For CJK,
# Arabic, and other non-Latin scripts, actual token counts may be 2-4x higher
# than estimated (these scripts typically have ~1-2 chars/token). Consider using
# actual tokenizer counts for multilingual content approaching token limits.
DEFAULT_CHARS_PER_TOKEN = 4


def estimate_tokens(
    text: str,
    chars_per_token: float = DEFAULT_CHARS_PER_TOKEN,
    tokenizer: Tokenizer | None = None,
) -> int:
    """Estimate token count from text.

    Uses the provided tokenizer for accurate counting, or falls back to
    character-based heuristic if no tokenizer is provided.

    Args:
        text: Text to estimate tokens for.
        chars_per_token: Average characters per token (default 4, used when no tokenizer).
        tokenizer: Optional tokenizer function for accurate counting.

    Returns:
        Token count (exact if tokenizer provided, estimated otherwise).
    """
    if not text:
        return 0
    if tokenizer is not None:
        return tokenizer(text)
    return max(1, int(len(text) / chars_per_token))


def estimate_message_tokens(
    role: str,
    content: str,
    tool_calls: list[ToolCall] | None = None,
    tokenizer: Tokenizer | None = None,
) -> int:
    """Estimate tokens for a single message including role overhead.

    Chat models add overhead for message structure (role, delimiters).
    This estimates ~4 tokens of overhead per message. Tool-call payloads
    (function names, serialized arguments) are included when present.

    Args:
        role: Message role (system, user, assistant, tool).
        content: Message content.
        tool_calls: Tool calls attached to the message (assistant messages).
        tokenizer: Optional tokenizer function for accurate counting.

    Returns:
        Token count including overhead (exact if tokenizer provided).
    """
    # ~4 tokens overhead for role + message delimiters
    overhead = 4
    tokens = overhead + estimate_tokens(content, tokenizer=tokenizer)
    if tool_calls:
        serialized = json.dumps(
            [dataclasses.asdict(tc) for tc in tool_calls], separators=(",", ":")
        )
        tokens += estimate_tokens(serialized, tokenizer=tokenizer)
    return tokens
