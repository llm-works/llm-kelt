"""Base backend protocol and types."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal


@dataclass
class Message:
    """A chat message."""

    role: Literal["system", "user", "assistant"]
    content: str


@dataclass
class ChatResponse:
    """Response from a chat completion."""

    content: str
    model: str
    usage: dict | None = None
    finish_reason: str | None = None


class Backend(ABC):
    """Abstract base class for LLM backends."""

    @abstractmethod
    async def chat(
        self,
        messages: list[Message],
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs,
    ) -> ChatResponse:
        """
        Send a chat completion request.

        Args:
            messages: List of messages in the conversation
            system: System prompt (injected appropriately for each backend)
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            **kwargs: Additional backend-specific parameters

        Returns:
            ChatResponse with the model's response
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close any open connections."""
        pass
