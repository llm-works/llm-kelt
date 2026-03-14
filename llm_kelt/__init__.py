"""llm-kelt: Framework for collecting and managing LLM memory.

Provides tools for storing facts, feedback, preferences, solutions, and other
signals that can be injected into LLM prompts or used for training.

Architecture:
    - kelt.atomic.* - Fact-based memory storage (assertions, solutions, feedback, etc.)
    - kelt.train.* - Training manifest and execution (DPO, SFT)
    - kelt.query - Context-aware LLM queries

Usage:
    from llm_kelt import ClientFactory, ClientContext
    from appinfra.config import Config
    from appinfra.log import LoggerFactory, LogConfig

    # Initialize
    config = Config("etc/llm-kelt.yaml")
    lg = LoggerFactory.create_root(LogConfig.from_params(level="info"))
    factory = ClientFactory(lg)

    # Create client with isolation context
    context = ClientContext(context_key="my-agent")
    kelt = factory.create_from_config(context=context, config=config)

    # Access atomic memory primitives via kelt.atomic.*
    kelt.atomic.assertions.add("Timezone: UTC", category="settings")
    kelt.atomic.solutions.record(
        agent_name="code-reviewer",
        problem="Review PR #123",
        problem_context={"messages": [...]},
        answer={"verdict": "approved"},
        tokens_used=1500,
        latency_ms=2340,
    )
    kelt.atomic.feedback.record(signal="positive", content_id=456)
    kelt.atomic.preferences.record(
        context="Summarize this",
        chosen="Concise version",
        rejected="Verbose version",
    )

    # Training via kelt.train.manifest.*
    manifest = kelt.train.manifest.create(
        key="my-adapter",
        method="dpo",
        model="Qwen/Qwen2.5-7B-Instruct",
        data=[{"prompt": "...", "chosen": "...", "rejected": "..."}],
    )
    kelt.train.manifest.submit(manifest)
    result = kelt.train.dpo.train(manifest)
"""

from .client import Client
from .core.errors import (
    ConfigError,
    ConflictError,
    DatabaseError,
    KeltError,
    NotFoundError,
    SchemaVersionError,
    ValidationError,
)
from .factory import ClientFactory
from .memory import ClientContext

__all__ = [
    "Client",
    "ClientFactory",
    "KeltError",
    "ValidationError",
    "NotFoundError",
    "DatabaseError",
    "ConfigError",
    "ConflictError",
    "SchemaVersionError",
    "ClientContext",
]
