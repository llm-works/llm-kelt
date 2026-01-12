"""High-level context-aware query interface."""

from dataclasses import dataclass, field

from .backends import Message
from .client import LLMClient
from .context import ContextBuilder


@dataclass
class Conversation:
    """
    A conversation with message history.

    Tracks messages for multi-turn conversations.
    """

    messages: list[Message] = field(default_factory=list)

    def add_user(self, content: str) -> None:
        """Add a user message."""
        self.messages.append(Message(role="user", content=content))

    def add_assistant(self, content: str) -> None:
        """Add an assistant message."""
        self.messages.append(Message(role="assistant", content=content))

    def clear(self) -> None:
        """Clear conversation history."""
        self.messages.clear()


class ContextQuery:
    """
    High-level interface for making context-aware queries.

    Combines the LLM client with context building to provide
    a simple API for context-aware AI interactions.

    Usage:
        query = ContextQuery(client, context_builder)

        # Single question
        response = await query.ask("What's a good approach to X?")

        # With conversation history
        conv = Conversation()
        response = await query.ask("Hello!", conversation=conv)
        response = await query.ask("Tell me more", conversation=conv)
    """

    def __init__(
        self,
        client: LLMClient,
        context_builder: ContextBuilder,
        base_system_prompt: str = "",
        temperature: float = 0.7,
    ):
        """
        Initialize context-aware query interface.

        Args:
            client: The LLM client to use
            context_builder: Context builder for fact injection
            base_system_prompt: Default system prompt to use
            temperature: Default temperature for responses
        """
        self._client = client
        self._context_builder = context_builder
        self._base_system_prompt = base_system_prompt
        self._temperature = temperature

    async def ask(  # cq: max-lines=40
        self,
        question: str,
        conversation: Conversation | None = None,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        include_facts: bool = True,
        fact_categories: list[str] | None = None,
    ) -> str:
        """
        Ask a question with context-aware context.

        Args:
            question: The user's question
            conversation: Optional conversation for multi-turn (will be updated)
            system_prompt: Override base system prompt (None = use default)
            temperature: Override temperature (None = use default)
            max_tokens: Maximum response tokens
            include_facts: Whether to inject user facts
            fact_categories: Only include facts from these categories

        Returns:
            The assistant's response

        Example:
            # Single question
            response = await query.ask("Explain gradient descent")

            # Multi-turn conversation
            conv = Conversation()
            r1 = await query.ask("What is ML?", conversation=conv)
            r2 = await query.ask("Tell me more about neural networks", conversation=conv)
        """
        # Build system prompt
        base = system_prompt if system_prompt is not None else self._base_system_prompt

        if include_facts:
            system = self._context_builder.build_system_prompt(
                base_prompt=base,
                categories=fact_categories,
            )
        else:
            system = base

        # Build messages
        if conversation:
            # Add user message to conversation
            conversation.add_user(question)
            messages = conversation.messages
        else:
            # Single message
            messages = [Message(role="user", content=question)]

        # Get response
        response = await self._client.chat(
            messages=messages,
            system=system if system else None,
            temperature=temperature if temperature is not None else self._temperature,
            max_tokens=max_tokens,
        )

        # Update conversation if provided
        if conversation:
            conversation.add_assistant(response)

        return response

    async def ask_without_facts(
        self,
        question: str,
        conversation: Conversation | None = None,
        system_prompt: str | None = None,
        **kwargs,
    ) -> str:
        """
        Ask a question without injecting user facts.

        Convenience method for queries that shouldn't be context-aware.

        Args:
            question: The user's question
            conversation: Optional conversation
            system_prompt: Optional system prompt
            **kwargs: Additional arguments passed to ask()

        Returns:
            The assistant's response
        """
        return await self.ask(
            question=question,
            conversation=conversation,
            system_prompt=system_prompt,
            include_facts=False,
            **kwargs,
        )

    def get_injected_context(
        self,
        system_prompt: str | None = None,
        fact_categories: list[str] | None = None,
    ) -> str:
        """
        Preview the system prompt that would be sent.

        Useful for debugging and understanding what context is injected.

        Args:
            system_prompt: Base system prompt (None = use default)
            fact_categories: Only include facts from these categories

        Returns:
            The complete system prompt with facts
        """
        base = system_prompt if system_prompt is not None else self._base_system_prompt
        return self._context_builder.build_system_prompt(
            base_prompt=base,
            categories=fact_categories,
        )

    async def close(self) -> None:
        """Close the underlying LLM client."""
        await self._client.close()

    async def __aenter__(self) -> "ContextQuery":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
