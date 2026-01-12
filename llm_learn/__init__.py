"""llm-learn: Framework for collecting and managing LLM context.

Provides tools for storing facts, feedback, preferences, and other signals
that can be injected into LLM prompts.

Usage:
    from llm_learn import LearnClient

    learn = LearnClient(profile_id=1)

    # Add facts for context injection
    learn.facts.add("Timezone: UTC", category="settings")

    # Record feedback
    learn.feedback.record(
        content_text="Response content...",
        signal="positive",
    )

    # Record preference pair
    learn.preferences.record(
        context="Summarize this",
        chosen="Concise version",
        rejected="Verbose version",
    )
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
