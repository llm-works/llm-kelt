"""Token estimation utilities for conversation management.

Provides fast, approximate token counting without requiring tokenizer dependencies.
Uses character-based heuristics calibrated for common LLM tokenizers.
"""

from __future__ import annotations

# Average chars per token varies by model/tokenizer, but 4 is a reasonable default.
# GPT-style tokenizers: ~4 chars/token for English text.
#
# LIMITATION: This heuristic is calibrated for English/Latin scripts. For CJK,
# Arabic, and other non-Latin scripts, actual token counts may be 2-4x higher
# than estimated (these scripts typically have ~1-2 chars/token). Consider using
# actual tokenizer counts for multilingual content approaching token limits.
DEFAULT_CHARS_PER_TOKEN = 4


def estimate_tokens(text: str, chars_per_token: float = DEFAULT_CHARS_PER_TOKEN) -> int:
    """Estimate token count from text using character heuristic.

    This is an approximation. For exact counts, use the model's tokenizer.

    Args:
        text: Text to estimate tokens for.
        chars_per_token: Average characters per token (default 4).

    Returns:
        Estimated token count.
    """
    if not text:
        return 0
    return max(1, int(len(text) / chars_per_token))


def estimate_message_tokens(role: str, content: str) -> int:
    """Estimate tokens for a single message including role overhead.

    Chat models add overhead for message structure (role, delimiters).
    This estimates ~4 tokens of overhead per message.

    Args:
        role: Message role (system, user, assistant, tool).
        content: Message content.

    Returns:
        Estimated token count including overhead.
    """
    # ~4 tokens overhead for role + message delimiters
    overhead = 4
    return overhead + estimate_tokens(content)
