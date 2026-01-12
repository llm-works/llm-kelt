"""OpenAI-compatible backend for local inference and OpenAI API."""

import os

import httpx

from .base import (
    Backend,
    BackendRequestError,
    BackendTimeoutError,
    BackendUnavailableError,
    ChatResponse,
    Message,
)


class OpenAICompatibleBackend(Backend):
    """
    Backend for OpenAI-compatible APIs.

    Works with:
    - Local inference server (infer) at localhost
    - OpenAI API
    - Any OpenAI-compatible endpoint (vLLM, LMStudio, Ollama, etc.)
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout: float = 120.0,
    ):
        """
        Initialize OpenAI-compatible backend.

        Args:
            base_url: API base URL (e.g., "http://localhost:8000/v1")
            model: Model name to use
            api_key: API key (optional for local, required for OpenAI)
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

        # Get API key from param or environment
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")

        # Build headers
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=headers,
            timeout=timeout,
        )

    async def chat(  # cq: max-lines=45
        self,
        messages: list[Message],
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs,
    ) -> ChatResponse:
        """Send chat completion request to OpenAI-compatible endpoint."""
        # Build messages list
        api_messages = []

        # Add system message if provided
        if system:
            api_messages.append({"role": "system", "content": system})

        # Add conversation messages
        for msg in messages:
            api_messages.append({"role": msg.role, "content": msg.content})

        # Build request payload
        payload = {
            "model": self.model,
            "messages": api_messages,
            "temperature": temperature,
        }

        if max_tokens:
            payload["max_tokens"] = max_tokens

        # Add any extra kwargs
        payload.update(kwargs)

        # Make request with exception translation
        try:
            response = await self._client.post("/chat/completions", json=payload)
            response.raise_for_status()
        except httpx.ConnectError as e:
            raise BackendUnavailableError(f"Cannot connect to {self.base_url}: {e}") from e
        except httpx.TimeoutException as e:
            raise BackendTimeoutError(f"Request timed out after {self.timeout}s") from e
        except httpx.HTTPStatusError as e:
            raise BackendRequestError(e.response.status_code, e.response.text[:500]) from e

        data = response.json()

        # Extract response
        choice = data["choices"][0]
        return ChatResponse(
            content=choice["message"]["content"],
            model=data.get("model", self.model),
            usage=data.get("usage"),
            finish_reason=choice.get("finish_reason"),
        )

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()

    @classmethod
    def from_config(cls, config: dict) -> "OpenAICompatibleBackend":
        """
        Create backend from configuration dict.

        Args:
            config: Dict with base_url, model, and optional api_key

        Returns:
            Configured backend instance
        """
        return cls(
            base_url=config["base_url"],
            model=config["model"],
            api_key=config.get("api_key"),
            timeout=config.get("timeout", 120.0),
        )
