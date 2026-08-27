# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-kelt Authors

"""Abstract base classes for conversation compaction strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..session import Conversation


class Compactor(ABC):
    """Base class for synchronous conversation compaction strategies.

    When a conversation approaches its token limit, compactors reduce the
    context while preserving important information. Different strategies
    trade off between information loss and token savings.

    Use this base class for compaction strategies that don't perform I/O
    (e.g., sliding window, simple truncation).
    """

    @abstractmethod
    def compact(self, conversation: Conversation) -> None:
        """Compact the conversation in-place.

        Args:
            conversation: Conversation to compact.
        """
        ...


class AsyncCompactor(ABC):
    """Base class for asynchronous conversation compaction strategies.

    Use this base class for compaction strategies that perform I/O operations
    (e.g., LLM-based summarization). Async compactors avoid blocking the event
    loop during long-running operations.

    When using an AsyncCompactor, callers must use ``append_async()`` instead
    of ``append()`` on the Conversation.
    """

    @abstractmethod
    async def compact(self, conversation: Conversation) -> None:
        """Compact the conversation in-place.

        Args:
            conversation: Conversation to compact.
        """
        ...
