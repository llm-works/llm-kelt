"""Conversation layer — stateful dialogue with compaction and persistence.

Provides the primitives for multi-turn, context-aware conversations:

- **types**: Re-exports Message, ToolCall, Role from llm-saia
- **session**: Conversation management with token tracking
- **compaction**: Strategies for reducing conversation size (window, summarizing)
- **storage**: Session persistence backends (file, database)

Usage::

    from llm_kelt.conversation import Conversation, Config, Message
    from llm_kelt.conversation.storage import FileSessionStorage
    from llm_kelt.conversation.compaction import SlidingWindowCompactor

    # Create conversation
    conv = Conversation(config=Config(max_tokens=32000))
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

from .compaction import Compactor, SlidingWindowCompactor, SummarizingCompactor
from .session import Config, Conversation
from .storage import FileSessionStorage, SessionStorage, SessionSummary, StoredSession
from .tokens import estimate_message_tokens, estimate_tokens
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
    "Compactor",
    "SlidingWindowCompactor",
    "SummarizingCompactor",
    # Storage
    "SessionStorage",
    "StoredSession",
    "SessionSummary",
    "FileSessionStorage",
    # Tokens
    "estimate_tokens",
    "estimate_message_tokens",
]
