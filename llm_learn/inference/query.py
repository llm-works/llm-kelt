"""High-level context-aware query interface."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from llm_infer.client import LLMClient

from .context import ContextBuilder

if TYPE_CHECKING:
    from .embedder import Embedder


@dataclass
class RAGArgs:
    """Configuration for RAG-based fact retrieval."""

    top_k: int = 10
    min_similarity: float = 0.3
    model_name: str | None = None  # Defaults to embedder's model
    categories: list[str] | None = None  # Filter results to these categories


@dataclass
class Conversation:
    """
    A conversation with message history.

    Tracks messages for multi-turn conversations.
    """

    messages: list[dict[str, str]] = field(default_factory=list)

    def add_user(self, content: str) -> None:
        """Add a user message."""
        self.messages.append({"role": "user", "content": content})

    def add_assistant(self, content: str) -> None:
        """Add an assistant message."""
        self.messages.append({"role": "assistant", "content": content})

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
        embedder: "Embedder | None" = None,
    ):
        """
        Initialize context-aware query interface.

        Args:
            client: The LLM client to use
            context_builder: Context builder for fact injection
            base_system_prompt: Default system prompt to use
            temperature: Default temperature for responses
            embedder: Optional embedder for RAG-based fact retrieval
        """
        self._client = client
        self._context_builder = context_builder
        self._base_system_prompt = base_system_prompt
        self._temperature = temperature
        self._embedder = embedder

    async def ask(  # cq: max-lines=50
        self,
        question: str,
        conversation: Conversation | None = None,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        include_facts: bool = True,
        fact_categories: list[str] | None = None,
        rag: RAGArgs | None = None,
    ) -> str:
        """
        Ask a question with context-aware context.

        Args:
            question: The user's question
            conversation: Optional conversation for multi-turn (will be updated)
            system_prompt: Override base system prompt (None = use default)
            temperature: Override temperature (None = use default)
            max_tokens: Maximum response tokens
            include_facts: Whether to inject user facts (ignored when rag is provided)
            fact_categories: Only include facts from these categories (ignored when rag
                is provided; use rag.categories instead)
            rag: RAG configuration - enables semantic retrieval when provided

        Returns:
            The assistant's response

        Raises:
            ValueError: If rag is provided but no embedder is configured

        Example:
            # Single question
            response = await query.ask("Explain gradient descent")

            # Multi-turn conversation
            conv = Conversation()
            r1 = await query.ask("What is ML?", conversation=conv)
            r2 = await query.ask("Tell me more about neural networks", conversation=conv)

            # With RAG
            response = await query.ask("What's my preferred coding style?", rag=RAGArgs())
        """
        # Build system prompt
        base = system_prompt if system_prompt is not None else self._base_system_prompt

        if rag is not None:
            system = await self._build_rag_prompt(base, question, rag)
        elif include_facts:
            system = self._context_builder.build_system_prompt(
                base_prompt=base,
                categories=fact_categories,
            )
        else:
            system = base

        # Build messages
        messages = self._build_messages(question, conversation)

        # Get response
        response = await self._client.chat_async(
            messages=messages,
            system=system if system else None,
            temperature=temperature if temperature is not None else self._temperature,
            max_tokens=max_tokens,
        )

        # Update conversation if provided
        if conversation:
            conversation.add_assistant(response)

        return response

    async def _build_rag_prompt(
        self,
        base: str,
        question: str,
        rag: RAGArgs,
    ) -> str:
        """Build system prompt using RAG-based fact retrieval."""
        if self._embedder is None:
            raise ValueError("RAG requires embedder to be configured")

        # Embed the question
        result = await self._embedder.embed_async(question)

        # Determine model name for search
        model_name = rag.model_name if rag.model_name else self._embedder.model

        # Search for similar facts (category filtering done in SQL for efficiency)
        scored_facts = self._context_builder.facts_client.search_similar(
            embedding=result.embedding,
            model_name=model_name,
            top_k=rag.top_k,
            min_similarity=rag.min_similarity,
            categories=rag.categories,
        )

        # Extract facts from scored results
        facts = [sf.fact for sf in scored_facts]

        # Build prompt with retrieved facts
        return self._context_builder.build_system_prompt_from_facts(base, facts)

    def _build_messages(
        self,
        question: str,
        conversation: Conversation | None,
    ) -> list[dict[str, str]]:
        """Build message list, updating conversation if provided."""
        if conversation:
            conversation.add_user(question)
            return conversation.messages
        return [{"role": "user", "content": question}]

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
        await self._client.aclose()

    async def __aenter__(self) -> "ContextQuery":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
