"""llm-learn: Framework for collecting and managing LLM memory.

Provides tools for storing facts, feedback, preferences, solutions, and other
signals that can be injected into LLM prompts or used for training.

Uses a unified fact-based memory architecture (memory v1) where all knowledge
is stored in a base facts table with type-specific detail tables.

Usage:
    from llm_learn import LearnClient

    learn = LearnClient(profile_id=1)

    # Access memory v1 primitives via learn.v1.*
    learn.v1.assertions.add("Timezone: UTC", category="settings")
    learn.v1.solutions.record(
        agent_name="code-reviewer",
        problem="Review PR #123",
        problem_context={"messages": [...]},
        answer={"verdict": "approved"},
        tokens_used=1500,
        latency_ms=2340,
    )
    learn.v1.feedback.record(signal="positive", content_id=456)
    learn.v1.preferences.record(
        context="Summarize this",
        chosen="Concise version",
        rejected="Verbose version",
    )

    # Convenience shorthand (equivalent to learn.v1.*)
    learn.assertions.add(...)
    learn.solutions.record(...)
"""

from .client import LearnClient
from .core.exceptions import (
    ConfigurationError,
    DatabaseError,
    LearnError,
    NotFoundError,
    ValidationError,
)

__all__ = [
    "LearnClient",
    "LearnError",
    "ValidationError",
    "NotFoundError",
    "DatabaseError",
    "ConfigurationError",
]
