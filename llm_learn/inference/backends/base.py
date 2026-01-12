"""Base backend protocol and types."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

# --- Backend Exceptions ---
# These provide a transport-agnostic interface for error handling.
# Each backend translates its specific exceptions (httpx, anthropic SDK, etc.)
# into these semantic exceptions.


class BackendError(Exception):
    """Base exception for all backend errors."""

    pass


class BackendUnavailableError(BackendError):
    """Backend server is not reachable (connection refused, DNS failure, etc.)."""

    pass


class BackendTimeoutError(BackendError):
    """Request to backend timed out."""

    pass


class BackendRequestError(BackendError):
    """Backend returned an HTTP error status."""

    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"HTTP {status_code}: {detail}")


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
