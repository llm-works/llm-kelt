"""Serving layer for llm-kelt proxy server.

Provides an OpenAI-compatible API that injects kelt context
(facts, directives, etc.) into LLM requests.
"""

from .app import create_server, create_server_from_config

__all__ = ["create_server", "create_server_from_config"]
