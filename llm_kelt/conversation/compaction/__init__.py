"""Conversation compaction strategies.

Provides compactors that reduce conversation size while preserving
important information when approaching token limits.
"""

from .base import Compactor
from .summary import SummarizingCompactor
from .window import SlidingWindowCompactor

__all__ = ["Compactor", "SlidingWindowCompactor", "SummarizingCompactor"]
