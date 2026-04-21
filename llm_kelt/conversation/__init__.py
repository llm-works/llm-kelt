"""Conversation layer — stateful dialogue with compaction and persistence.

Provides the primitives for multi-turn, context-aware conversations:

- **types**: Re-exports Message, ToolCall, Role from llm-saia
- **session**: Conversation management with token tracking
- **compaction**: Strategies for reducing conversation size (window, summarizing)
- **storage**: Session persistence backends (file, database)

Usage::

    from appinfra.log import Logger
    from llm_kelt.conversation import Conversation, Config, Message
    from llm_kelt.conversation.storage import FileSessionStorage
    from llm_kelt.conversation.compaction import SlidingWindowCompactor

    # Create conversation (lg is a Logger instance)
    conv = Conversation(lg, config=Config(max_tokens=32000))
    conv.add("What happened in the news today?")
    conv.add("Here are today's top stories...", Role.ASSISTANT)

    # Persist
    storage = FileSessionStorage(lg, "~/.my-agent/sessions")
    storage.save("session-123", conv, metadata={"model": "qwen2.5-7b"})

    # Compact when needed
    if conv.needs_compaction():
        compactor = SlidingWindowCompactor()
        compactor.compact(conv)
"""

from ..core.errors import ContextOverflowError
from .compaction import AsyncCompactor, Compactor, SlidingWindowCompactor, SummarizingCompactor
from .session import Config, Conversation
from .storage import FileSessionStorage, SessionStorage, SessionSummary, StoredSession
from .tokens import Tokenizer, estimate_message_tokens, estimate_tokens
from .types import Message, Role, ToolCall

__all__ = [
    # Types
    "Message",
    "Role",
    "ToolCall",
    # Session
    "Conversation",
    "Config",
    # Compaction
    "AsyncCompactor",
    "Compactor",
    "SlidingWindowCompactor",
    "SummarizingCompactor",
    # Storage
    "SessionStorage",
    "StoredSession",
    "SessionSummary",
    "FileSessionStorage",
    # Tokens
    "Tokenizer",
    "estimate_tokens",
    "estimate_message_tokens",
    # Errors
    "ContextOverflowError",
]
