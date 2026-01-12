"""Anthropic (Claude) backend."""

import os

from .base import (
    Backend,
    BackendRequestError,
    BackendTimeoutError,
    BackendUnavailableError,
    ChatResponse,
    Message,
)


class AnthropicBackend(Backend):
    """
    Backend for Anthropic's Claude API.

    Requires the `anthropic` package to be installed.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: str | None = None,
        max_tokens: int = 4096,
    ):
        """
        Initialize Anthropic backend.

        Args:
            model: Model name (e.g., "claude-sonnet-4-20250514")
            api_key: API key (or set ANTHROPIC_API_KEY env var)
            max_tokens: Default max tokens for responses
        """
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "anthropic package is required for AnthropicBackend. "
                "Install with: pip install anthropic"
            )

        self.model = model
        self.default_max_tokens = max_tokens

        # Get API key from param or environment
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError(
                "Anthropic API key required. Set ANTHROPIC_API_KEY env var "
                "or pass api_key parameter."
            )

        self._client = anthropic.AsyncAnthropic(api_key=key)

    async def chat(  # cq: max-lines=40
        self,
        messages: list[Message],
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs,
    ) -> ChatResponse:
        """Send chat completion request to Anthropic API."""
        # Build messages list (Anthropic format)
        api_messages: list[dict[str, str]] = []
        for msg in messages:
            if msg.role == "system":
                # Anthropic handles system separately, skip in messages
                continue
            api_messages.append({"role": msg.role, "content": msg.content})

        # Make request with exception translation
        # Import anthropic for exception types (module already loaded at __init__)
        import anthropic

        try:
            response = await self._client.messages.create(
                model=self.model,
                messages=api_messages,  # type: ignore[arg-type]
                temperature=temperature,
                max_tokens=max_tokens or self.default_max_tokens,
                system=system or "",
            )
        except anthropic.APIConnectionError as e:
            raise BackendUnavailableError(f"Cannot connect to Anthropic API: {e}") from e
        except anthropic.APITimeoutError as e:
            raise BackendTimeoutError("Anthropic API request timed out") from e
        except anthropic.APIStatusError as e:
            raise BackendRequestError(e.status_code, str(e)[:500]) from e

        # Extract content (Anthropic returns list of content blocks)
        content = ""
        for block in response.content:
            if block.type == "text":
                content += block.text

        return ChatResponse(
            content=content,
            model=response.model,
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
            finish_reason=response.stop_reason,
        )

    async def close(self) -> None:
        """Close the Anthropic client."""
        await self._client.close()

    @classmethod
    def from_config(cls, config: dict) -> "AnthropicBackend":
        """
        Create backend from configuration dict.

        Args:
            config: Dict with model and optional api_key, max_tokens

        Returns:
            Configured backend instance
        """
        return cls(
            model=config["model"],
            api_key=config.get("api_key"),
            max_tokens=config.get("max_tokens", 4096),
        )
