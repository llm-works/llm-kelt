"""Inference module for LLM interactions with context injection.

This module provides:
- ContextBuilder: Builds system prompts with injected facts
- ContextQuery: High-level interface for context-aware queries
- Embedder: Client for generating embeddings

For LLM client functionality, use llm_infer.client directly:
    from llm_infer.client import LLMClient

Usage:
    from llm_infer.client import LLMClient
    from llm_learn import LearnClient
    from llm_learn.inference import ContextBuilder

    # Setup
    learn = LearnClient(profile_id=1)
    client = LLMClient.from_config(config["llm"])
    context = ContextBuilder(learn.assertions)

    # Build prompt with facts injected
    system = context.build_system_prompt("You are a helpful assistant.")
    response = await client.chat_async(messages, system=system)
"""

from .context import ContextBuilder
from .embed_facts import EmbedFactsResult, embed_missing_facts
from .embedder import Embedder, EmbeddingResult
from .query import ContextQuery, Conversation, RAGArgs

__all__ = [
    "ContextBuilder",
    "ContextQuery",
    "Conversation",
    "Embedder",
    "RAGArgs",
    "embed_missing_facts",
    "EmbeddingResult",
    "EmbedFactsResult",
]
