"""llm-learn: Framework for collecting and managing LLM memory.

Provides tools for storing facts, feedback, preferences, solutions, and other
signals that can be injected into LLM prompts or used for training.

Architecture:
    - learn.atomic.* - Fact-based memory storage (assertions, solutions, feedback, etc.)
    - learn.train.* - Training methods (DPO, SFT, etc.)
    - learn.query - Context-aware LLM queries

Usage:
    from llm_learn import LearnClientFactory, IsolationContext
    from appinfra.config import Config
    from appinfra.log import LoggerFactory, LogConfig

    # Initialize
    config = Config("etc/llm-learn.yaml")
    lg = LoggerFactory.create_root(LogConfig.from_params(level="info"))
    factory = LearnClientFactory(lg)

    # Create client with isolation context
    context = IsolationContext(context_key="my-agent")
    learn = factory.create_from_config(context=context, config=config)

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

    # Access training methods via learn.train.*
    learn.train.dpo.create(adapter_name="my-adapter")
    learn.train.dpo.list_runs(status="pending")
"""

from .client import LearnClient
from .core.exceptions import (
    ConfigurationError,
    ConflictError,
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
    "ConflictError",
    "SchemaVersionError",
    "IsolationContext",
]
