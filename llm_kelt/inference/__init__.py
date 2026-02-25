"""Inference module for LLM interactions with context injection.

This module provides:
- ContextBuilder: Builds system prompts with injected facts
- ContextQuery: High-level interface for context-aware queries
- Embedder: Client for generating embeddings

For LLM client functionality, use llm_infer.client directly:
    from llm_infer.client import LLMClient

Usage:
    from llm_infer.client import LLMClient
    from llm_kelt import ClientFactory, ClientContext
    from llm_kelt.inference import ContextBuilder
    from appinfra.config import Config
    from appinfra.log import LoggerFactory, LogConfig

    # Setup
    config = Config("etc/llm-kelt.yaml")
    lg = LoggerFactory.create_root(LogConfig.from_params(level="info"))
    factory = ClientFactory(lg)
    context = ClientContext(context_key="my-agent")
    kelt = factory.create_from_config(context=context, config=config)
    client = LLMClient.from_config(config["llm"])
    context_builder = ContextBuilder(kelt.atomic.assertions)

    # Build prompt with facts injected
    system = context_builder.build_system_prompt("You are a helpful assistant.")
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
