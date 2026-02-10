"""llm-learn: Framework for collecting and managing LLM memory.

Provides tools for storing facts, feedback, preferences, solutions, and other
signals that can be injected into LLM prompts or used for training.

Architecture:
    - core: Domain, Workspace, Profile hierarchy with hash-based IDs
    - memory.atomic: Fact-based storage with type-specific detail tables

Usage:
    from llm_learn import LearnClient

    # Create client scoped to a profile (32-char hash ID)
    learn = LearnClient(profile_id="a3f8b2c1d4e5f6a7b8c9d0e1f2a3b4c5")

    # Access atomic memory primitives via learn.atomic.*
    learn.atomic.assertions.add("Timezone: UTC", category="settings")
    learn.atomic.solutions.record(
        agent_name="code-reviewer",
        problem="Review PR #123",
        problem_context={"messages": [...]},
        answer={"verdict": "approved"},
        tokens_used=1500,
        latency_ms=2340,
    )
    learn.atomic.feedback.record(signal="positive", content_id=456)
    learn.atomic.preferences.record(
        context="Summarize this",
        chosen="Concise version",
        rejected="Verbose version",
    )

    # Convenience shorthand (equivalent to learn.atomic.*)
    learn.assertions.add(...)
    learn.solutions.record(...)
"""

from .client import LearnClient
from .core.exceptions import (
    ConfigurationError,
    DatabaseError,
    LearnError,
    NotFoundError,
    SchemaVersionError,
    ValidationError,
)
from .factory import LearnClientFactory
from .memory import IsolationContext

__all__ = [
    "LearnClient",
    "LearnClientFactory",
    "LearnError",
    "ValidationError",
    "NotFoundError",
    "DatabaseError",
    "ConfigurationError",
    "SchemaVersionError",
    "IsolationContext",
]
