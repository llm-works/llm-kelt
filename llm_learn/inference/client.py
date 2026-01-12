"""Unified LLM client that supports multiple backends."""

from .backends import AnthropicBackend, Backend, Message, OpenAICompatibleBackend
from .backends.base import ChatResponse


class LLMClient:
    """
    Unified LLM client that works with any configured backend.

    Supports:
    - Local inference via OpenAI-compatible API
    - OpenAI API
    - Anthropic (Claude) API

    Usage:
        client = LLMClient.from_config(config["llm"])
        response = await client.chat([Message("user", "Hello!")])
    """

    def __init__(self, backend: Backend):
        """
        Initialize client with a backend.

        Args:
            backend: The LLM backend to use
        """
        self._backend = backend

    async def chat(
        self,
        messages: list[Message] | list[dict],
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs,
    ) -> str:
        """
        Send a chat completion request and return the response content.

        Args:
            messages: List of messages (Message objects or dicts with role/content)
            system: System prompt to prepend
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens in response
            **kwargs: Additional backend-specific parameters

        Returns:
            The assistant's response content as a string
        """
        # Convert dicts to Message objects if needed
        msg_objects = []
        for msg in messages:
            if isinstance(msg, Message):
                msg_objects.append(msg)
            elif isinstance(msg, dict):
                msg_objects.append(Message(role=msg["role"], content=msg["content"]))
            else:
                raise TypeError(f"Expected Message or dict, got {type(msg)}")

        response = await self._backend.chat(
            messages=msg_objects,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        return response.content

    async def chat_full(
        self,
        messages: list[Message] | list[dict],
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs,
    ) -> ChatResponse:
        """
        Send a chat completion request and return the full response.

        Same as chat() but returns ChatResponse with metadata.

        Args:
            messages: List of messages
            system: System prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            **kwargs: Additional parameters

        Returns:
            ChatResponse with content, model, usage, and finish_reason
        """
        msg_objects = []
        for msg in messages:
            if isinstance(msg, Message):
                msg_objects.append(msg)
            elif isinstance(msg, dict):
                msg_objects.append(Message(role=msg["role"], content=msg["content"]))
            else:
                raise TypeError(f"Expected Message or dict, got {type(msg)}")

        return await self._backend.chat(
            messages=msg_objects,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

    async def close(self) -> None:
        """Close the underlying backend connection."""
        await self._backend.close()

    async def __aenter__(self) -> "LLMClient":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()

    @classmethod
    def from_config(cls, config: dict) -> "LLMClient":
        """
        Create client from configuration.

        Args:
            config: LLM configuration dict with 'default' and 'backends' keys

        Returns:
            Configured LLMClient

        Example config:
            llm:
              default: anthropic
              backends:
                local:
                  type: openai_compatible
                  base_url: http://localhost:8000/v1
                  model: qwen2.5-72b
                anthropic:
                  type: anthropic
                  model: claude-sonnet-4-20250514
                openai:
                  type: openai
                  base_url: https://api.openai.com/v1
                  model: gpt-4o
        """
        default_backend = config.get("default", "local")
        backends_config = config.get("backends", {})

        if default_backend not in backends_config:
            available = list(backends_config.keys())
            raise ValueError(
                f"Default backend '{default_backend}' not found in config. Available: {available}"
            )

        backend_config = backends_config[default_backend]
        backend = cls._create_backend(backend_config)
        return cls(backend)

    @classmethod
    def from_backend_config(cls, backend_config: dict) -> "LLMClient":
        """
        Create client from a single backend configuration.

        Args:
            backend_config: Backend configuration dict with 'type' key

        Returns:
            Configured LLMClient
        """
        backend = cls._create_backend(backend_config)
        return cls(backend)

    @staticmethod
    def _create_backend(config: dict) -> Backend:
        """Create a backend from configuration."""
        backend_type = config.get("type", "openai_compatible")

        if backend_type == "openai_compatible":
            return OpenAICompatibleBackend.from_config(config)
        elif backend_type == "openai":
            # OpenAI uses the same format as openai_compatible
            return OpenAICompatibleBackend.from_config(config)
        elif backend_type == "anthropic":
            return AnthropicBackend.from_config(config)
        else:
            raise ValueError(f"Unknown backend type: {backend_type}")
