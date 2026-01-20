"""Inference module for LLM interactions with context injection.

This module provides:
- LLMClient: Unified client for multiple LLM backends
- ContextBuilder: Builds system prompts with injected facts
- ContextQuery: High-level interface for context-aware queries

Usage:
    from llm_learn.core import Database
    from llm_learn.collection import FactsClient
    from llm_learn.inference import LLMClient, ContextBuilder

    # Setup
    db = Database(config)
    facts = FactsClient(db.session, profile_id=1)
    client = LLMClient.from_config(config["llm"])
    context = ContextBuilder(facts)

    # Add some facts
    facts.add("Output format: markdown", category="preferences")
    facts.add("Timezone: UTC", category="settings")

    # Build prompt with facts injected
    system = context.build_system_prompt("You are a helpful assistant.")
    response = await client.chat(messages, system=system)
"""

from .backends import Backend, Message
from .backends.base import ChatResponse
from .client import LLMClient
from .context import ContextBuilder
from .embed_facts import EmbedFactsResult, embed_missing_facts
from .embedder import Embedder, EmbeddingResult
from .query import ContextQuery, Conversation

__all__ = [
    # Core classes
    "LLMClient",
    "ContextBuilder",
    "ContextQuery",
    "Conversation",
    "Embedder",
    # Utilities
    "embed_missing_facts",
    # Types
    "Backend",
    "Message",
    "ChatResponse",
    "EmbeddingResult",
    "EmbedFactsResult",
]
