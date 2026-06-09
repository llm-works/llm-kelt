"""Conversation compaction strategies.

Provides compactors that reduce conversation size while preserving
important information when approaching token limits.

Supports guards for validating compaction quality with retry/escalation::

    from llm_kelt.conversation.compaction import (
        SummarizingCompactor,
        token_reduction,
        preserve_keywords,
    )

    compactor = SummarizingCompactor(
        client=async_llm_client,
        guards=[token_reduction(min_ratio=0.5), preserve_keywords(["API", "error"])],
    )
"""

from .base import AsyncCompactor, Compactor
from .guard import (
    CompactionContext,
    CompactionGuard,
    CompactionGuardError,
    max_summary_tokens,
    preserve_keywords,
    token_reduction,
)
from .summary import SummarizingCompactor
from .tiered import (
    DEFAULT_TRIMMABLE_TOOLS,
    AsyncTieredCompactor,
    TieredCompactor,
)
from .window import SlidingWindowCompactor

__all__ = [
    # Base
    "AsyncCompactor",
    "Compactor",
    # Compactors
    "SlidingWindowCompactor",
    "SummarizingCompactor",
    "TieredCompactor",
    "AsyncTieredCompactor",
    "DEFAULT_TRIMMABLE_TOOLS",
    # Guards
    "CompactionContext",
    "CompactionGuard",
    "CompactionGuardError",
    "max_summary_tokens",
    "preserve_keywords",
    "token_reduction",
]
