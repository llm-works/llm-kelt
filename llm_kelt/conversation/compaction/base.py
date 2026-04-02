"""Abstract base class for conversation compaction strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..session import Conversation


class Compactor(ABC):
    """Base class for conversation compaction strategies.

    When a conversation approaches its token limit, compactors reduce the
    context while preserving important information. Different strategies
    trade off between information loss and token savings.
    """

    @abstractmethod
    def compact(self, conversation: Conversation) -> None:
        """Compact the conversation in-place.

        Args:
            conversation: Conversation to compact.
        """
        ...
