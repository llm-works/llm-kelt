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

    def _build_api_messages(self, messages: list[Message]) -> list[dict[str, str]]:
        """Convert messages to Anthropic format, filtering out system messages."""
        return [
            {"role": msg.role, "content": msg.content} for msg in messages if msg.role != "system"
        ]

    def _parse_response(self, response) -> ChatResponse:
        """Extract content and metadata from Anthropic API response."""
        content = "".join(block.text for block in response.content if block.type == "text")
        return ChatResponse(
            content=content,
            model=response.model,
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
            finish_reason=response.stop_reason,
        )

    async def chat(
        self,
        messages: list[Message],
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs,
    ) -> ChatResponse:
        """Send chat completion request to Anthropic API."""
        import anthropic

        api_messages = self._build_api_messages(messages)
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

        return self._parse_response(response)

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
