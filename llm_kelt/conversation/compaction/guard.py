# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-kelt Authors

"""Guards for validating compaction results with retry capability.

Adopts the guard pattern from saia for compaction quality assurance.
Guards validate the compaction result and can trigger retries with
escalating feedback to the summarizer LLM.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from ..types import Message

__all__ = [
    "CompactionContext",
    "CompactionGuard",
    "CompactionGuardError",
    "max_summary_tokens",
    "preserve_keywords",
    "token_reduction",
]


@dataclass
class CompactionContext:
    """Context passed to compaction guard validators.

    Provides access to before/after state for validation.

    Attributes:
        before_messages: Messages before compaction.
        after_messages: Messages after compaction.
        before_tokens: Token count before compaction.
        after_tokens: Token count after compaction (includes preserved messages).
        summary: The generated summary text (for summarizing compactor).
        summary_tokens: Token count of just the summary (for summarizing compactor).
        attempt: Current attempt number (0-indexed).
    """

    before_messages: list[Message]
    after_messages: list[Message]
    before_tokens: int
    after_tokens: int
    summary: str | None
    summary_tokens: int | None
    attempt: int

    @property
    def reduction_ratio(self) -> float:
        """Fraction of tokens saved (0.0 to 1.0)."""
        if self.before_tokens == 0:
            return 0.0
        return 1.0 - (self.after_tokens / self.before_tokens)


@dataclass(frozen=True)
class CompactionGuard:
    """Validates compaction result and retries with instruction on failure.

    Args:
        validator: Function receiving CompactionContext. Returns None if valid,
            error string if invalid.
        retry_instruction: Static string or callable ``(attempt, ctx, error) -> str``
            sent to the summarizer LLM on retry. Callables enable escalating tone.
        max_retries: Max retry attempts (must be >= 0). Default 2.
        name: Optional name for logging/debugging.

    Example:
        >>> guard = CompactionGuard(
        ...     validator=lambda ctx: "Too long" if ctx.after_tokens > 1000 else None,
        ...     retry_instruction="Make the summary shorter.",
        ...     name="length_check",
        ... )
    """

    validator: Callable[[CompactionContext], str | None]
    retry_instruction: str | Callable[[int, CompactionContext, str], str]
    max_retries: int = 2
    name: str | None = None

    def __post_init__(self) -> None:
        """Validate max_retries is non-negative."""
        if self.max_retries < 0:
            raise ValueError(f"max_retries must be >= 0, got {self.max_retries}")

    def resolve_instruction(self, attempt: int, ctx: CompactionContext, error: str) -> str:
        """Resolve retry instruction for the given attempt."""
        if callable(self.retry_instruction):
            return self.retry_instruction(attempt, ctx, error)
        return self.retry_instruction


class CompactionGuardError(Exception):
    """Raised when compaction fails guard validation after all retries exhausted."""

    def __init__(self, guard_name: str | None, error: str, attempts: int) -> None:
        self.guard_name = guard_name
        self.error = error
        self.attempts = attempts
        name = guard_name or "guard"
        super().__init__(f"Compaction {name} failed after {attempts} attempts: {error}")


# --- Escalation helper ---


def _escalating(
    requirement: str, polite: str, forceful: str
) -> Callable[[int, CompactionContext, str], str]:
    """Create an escalating instruction with failure count feedback."""

    def instruction(attempt: int, ctx: CompactionContext, error: str) -> str:
        if attempt == 0:
            return polite
        return f"YOU HAVE FAILED TO {requirement.upper()} {attempt + 1} TIMES. {forceful}"

    return instruction


# --- Pre-built guards ---


def token_reduction(
    min_ratio: float = 0.3, max_retries: int = 2, *, escalate: bool = True
) -> CompactionGuard:
    """Require minimum token reduction ratio.

    Args:
        min_ratio: Minimum fraction of tokens to remove (0.0 to 1.0). Default 0.3 (30%).
        max_retries: Max retry attempts. Default 2.
        escalate: Use increasingly forceful retry instructions. Default True.

    Raises:
        ValueError: If min_ratio is not between 0.0 and 1.0.
    """
    if not 0.0 <= min_ratio <= 1.0:
        raise ValueError(f"min_ratio must be between 0.0 and 1.0, got {min_ratio}")

    def check(ctx: CompactionContext) -> str | None:
        if ctx.reduction_ratio < min_ratio:
            return (
                f"Reduction ratio {ctx.reduction_ratio:.1%} is below minimum {min_ratio:.1%}. "
                f"Before: {ctx.before_tokens}, after: {ctx.after_tokens}."
            )
        return None

    static = (
        f"Your summary is too long. Reduce it further to achieve at least {min_ratio:.0%} "
        f"token reduction. Focus only on the most important information."
    )
    return CompactionGuard(
        validator=check,
        retry_instruction=_escalating(
            f"reduce by {min_ratio:.0%}",
            static,
            f"REDUCE BY AT LEAST {min_ratio:.0%}. Cut everything non-essential. "
            f"Key facts only. No details, no examples, no elaboration. Do it NOW.",
        )
        if escalate
        else static,
        max_retries=max_retries,
        name="token_reduction",
    )


def preserve_keywords(
    keywords: list[str], max_retries: int = 2, *, escalate: bool = True
) -> CompactionGuard:
    """Require specific keywords to appear in the summary.

    Args:
        keywords: List of keywords that must appear in the compacted result.
        max_retries: Max retry attempts. Default 2.
        escalate: Use increasingly forceful retry instructions. Default True.
    """

    def check(ctx: CompactionContext) -> str | None:
        if ctx.summary is None:
            return None  # Not a summarizing compaction
        summary_lower = ctx.summary.lower()
        missing = [kw for kw in keywords if kw.lower() not in summary_lower]
        if missing:
            return f"Missing required keywords: {', '.join(missing)}"
        return None

    keywords_str = ", ".join(keywords)
    static = (
        f"Your summary is missing important keywords: {keywords_str}. "
        f"Include these terms in your summary."
    )
    return CompactionGuard(
        validator=check,
        retry_instruction=_escalating(
            "include required keywords",
            static,
            f"YOU MUST INCLUDE: {keywords_str}. These are CRITICAL terms. "
            f"Rewrite to include ALL of them. Do it NOW.",
        )
        if escalate
        else static,
        max_retries=max_retries,
        name="preserve_keywords",
    )


def max_summary_tokens(
    max_tokens: int, max_retries: int = 2, *, escalate: bool = True
) -> CompactionGuard:
    """Limit summary to maximum token count.

    Args:
        max_tokens: Maximum tokens allowed in the summary.
        max_retries: Max retry attempts. Default 2.
        escalate: Use increasingly forceful retry instructions. Default True.

    Raises:
        ValueError: If max_tokens is negative.
    """
    if max_tokens < 0:
        raise ValueError(f"max_tokens must be >= 0, got {max_tokens}")

    def check(ctx: CompactionContext) -> str | None:
        if ctx.summary_tokens is None:
            return None  # Not a summarizing compaction
        if ctx.summary_tokens > max_tokens:
            return f"Summary has {ctx.summary_tokens} tokens (max: {max_tokens})"
        return None

    def instruction(attempt: int, ctx: CompactionContext, error: str) -> str:
        tokens = ctx.summary_tokens or 0
        if attempt == 0:
            return (
                f"Your summary has {tokens} tokens but the limit is {max_tokens}. "
                f"Shorten it while preserving key information."
            )
        return (
            f"YOU HAVE FAILED TO STAY UNDER {max_tokens} TOKENS {attempt + 1} TIMES. "
            f"Current: {tokens}. Strip EVERYTHING non-essential. Do it NOW."
        )

    static = f"Your summary exceeds {max_tokens} tokens. Make it shorter."
    return CompactionGuard(
        validator=check,
        retry_instruction=instruction if escalate else static,
        max_retries=max_retries,
        name="max_summary_tokens",
    )
