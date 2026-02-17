"""Main LearnClient - primary API entry point."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from appinfra.dot_dict import DotDict
from appinfra.log import Logger
from llm_infer.client import ChatClient

from .core.content import ContentStore
from .core.database import Database
from .core.embedding import EmbeddingStore
from .core.exceptions import SchemaVersionError
from .core.schema import SchemaManager, SchemaState, SchemaStatus
from .inference.context import ContextBuilder
from .inference.embedder import Embedder
from .inference.query import ContextQuery
from .memory import atomic
from .memory.isolation import IsolationContext

if TYPE_CHECKING:
    from .memory.atomic import Protocol
    from .training import TrainClient


class LearnClient:
    """
    Main client for the Learn framework, scoped to an isolation context.

    Provides unified access to all framework capabilities:
    - learn.atomic.* - Fact-based memory storage (assertions, solutions, feedback, etc.)
    - learn.train.* - Training methods (DPO, SFT, etc.)
    - learn.query - Context-aware LLM queries

    Usage (via factory - recommended):
        from appinfra.config import Config
        from appinfra.log import LogConfig, LoggerFactory
        from llm_learn import LearnClientFactory, IsolationContext

        config = Config("etc/llm-learn.yaml")
        lg = LoggerFactory.create_root(LogConfig.from_params(level="info"))

        factory = LearnClientFactory(lg)
        context = IsolationContext(context_key="my-agent", schema_name="public")
        learn = factory.create_from_config(context=context, config=config)

    Usage (direct - for testing or shared resources):
        from llm_learn import LearnClient, IsolationContext

        context = IsolationContext(context_key="my-agent", schema_name="public")
        learn = LearnClient(
            database=db, context=context, lg=lg,
            embedder=embedder, llm_client=llm_client,
        )

    Memory API:
        learn.atomic.assertions.add("Prefers concise explanations", category="preferences")
        learn.atomic.solutions.record(agent_name="reviewer", problem="...", ...)
        learn.atomic.feedback.record(signal="positive", content_id=456)

    Training API:
        learn.train.dpo.create(adapter_name="my-adapter")
        learn.train.dpo.list_runs(status="pending")

    Query API (requires llm_client):
        response = await learn.query.ask("What's a good approach?")
    """

    def __init__(
        self,
        lg: Logger,
        database: Database,
        context: IsolationContext,
        embedder: Embedder | None = None,
        llm_client: ChatClient | None = None,
        learn_config: DotDict | None = None,
        ensure_schema: bool = True,
    ) -> None:
        """
        Initialize LearnClient with isolation context.

        Args:
            database: Database instance
            context: IsolationContext for data partitioning (any string format)
            lg: Optional logger instance
            embedder: Optional embedder for generating embeddings
            llm_client: Optional LLM client for context-aware queries
            learn_config: Optional learn settings
            ensure_schema: If True (default), auto-migrate schema on init
        """
        self._db = database
        self._context = context
        self._lg = lg
        self._embedder = embedder
        self._llm_client = llm_client
        self._learn_config = learn_config

        self._verify_schema(ensure=ensure_schema)
        self._setup_stores()
        self._setup_query_interface()

    def _setup_stores(self) -> None:
        """Initialize storage components."""
        self._embedding_store = EmbeddingStore(self._db.session)
        self._content = ContentStore(self._db.session, self._context.context_key)
        self._atomic = atomic.Protocol(
            self._lg,
            self._db.session,
            self._context.context_key,
            embedder=self._embedder,
            embedding_store=self._embedding_store,
        )
        self._context_builder = ContextBuilder(self._atomic.assertions)
        self._train: TrainClient | None = None

    def _setup_query_interface(self) -> None:
        """Initialize context query interface if LLM client is available."""
        self._context_query: ContextQuery | None = None
        if self._llm_client is not None:
            self._context_query = ContextQuery(
                client=self._llm_client,
                context_builder=self._context_builder,
                base_system_prompt=self._get_default_system_prompt(),
                embedder=self._embedder,
                embedding_adapter=self._atomic.embeddings,
            )

    @property
    def context(self) -> IsolationContext:
        """
        Get isolation context for this client.

        Returns the current IsolationContext with context_key and schema_name.
        """
        return self._context

    @property
    def context_key(self) -> str | None:
        """Get context key (convenience accessor for context.context_key)."""
        return self._context.context_key

    def with_isolation(
        self,
        *,
        context_key: str | None = ...,  # type: ignore[assignment]
        schema_name: str | None = ...,  # type: ignore[assignment]
        **kwargs: Any,
    ) -> LearnClient:
        """
        Return new client with isolation overrides.

        Only provided fields are overridden; omitted fields keep their current values.
        You can explicitly set a field to None to clear it.

        Args:
            context_key: New context_key, or None to clear. Omit to keep current.
            schema_name: New schema_name, or None to clear. Omit to keep current.
            **kwargs: Additional fields for extensibility

        Returns:
            New LearnClient with merged context

        Example:
            # Override just schema (keeps same context_key)
            learn.with_isolation(schema_name="public")

            # Clear context_key to None, keep schema_name
            learn.with_isolation(context_key=None)

            # Override both
            learn.with_isolation(
                context_key="other_context",
                schema_name="customer_other"
            )
        """
        from dataclasses import replace

        # Build overrides dict from provided kwargs
        valid_fields = {"context_key", "schema_name"}
        overrides = {k: v for k, v in kwargs.items() if k in valid_fields}

        # Add explicitly provided positional kwargs (type hints)
        if context_key is not ...:
            overrides["context_key"] = context_key
        if schema_name is not ...:
            overrides["schema_name"] = schema_name

        # Merge with current context
        merged = replace(self.context, **overrides)

        # Create new client with merged context
        return LearnClient(
            database=self._db,
            context=merged,
            lg=self._lg,
            embedder=self._embedder,
            llm_client=self._llm_client,
            learn_config=self._learn_config,
            ensure_schema=False,  # Don't re-run schema checks
        )

    @property
    def atomic(self) -> Protocol:
        """Access atomic memory protocol."""
        return self._atomic

    @property
    def content(self) -> ContentStore:
        """Access content storage API."""
        return self._content

    @property
    def train(self) -> TrainClient:
        """Access training client for DPO, SFT, etc."""
        if self._train is None:
            from .training import TrainClient

            self._train = TrainClient(
                self._lg,
                self._db.session,
                self._context.context_key,
            )
        return self._train

    @property
    def database(self) -> Database:
        """Access underlying database."""
        return self._db

    @property
    def llm_client(self) -> ChatClient | None:
        """Access underlying LLM client (None if not configured)."""
        return self._llm_client

    @property
    def embedder(self) -> Embedder | None:
        """Access underlying embedder (None if not configured)."""
        return self._embedder

    @property
    def learn_config(self) -> DotDict | None:
        """Access learn configuration (memory, embedding, default_system_prompt, etc.)."""
        return self._learn_config

    @property
    def context_builder(self) -> ContextBuilder:
        """Access context builder for prompt construction."""
        return self._context_builder

    @property
    def query(self) -> ContextQuery:
        """Access context-aware query interface.

        Raises:
            RuntimeError: If llm_client was not provided during init.
        """
        if self._context_query is None:
            raise RuntimeError("LLM client not configured. Pass llm_client to LearnClient.")
        return self._context_query

    def _get_default_system_prompt(self) -> str:
        """Get default system prompt from learn config."""
        if self._learn_config is None:
            return ""
        return getattr(self._learn_config, "default_system_prompt", "") or ""

    def _verify_schema(self, *, ensure: bool) -> None:
        """Verify database schema is current, optionally auto-migrating.

        Args:
            ensure: If True, create database and run migrations automatically.
                    If False, only verify — raise SchemaVersionError if not current.
        """
        if ensure:
            self._db.ensure_database()
            manager = SchemaManager(self._lg, self._db.engine)
            manager.ensure_schema()
            return

        manager = SchemaManager(self._lg, self._db.engine)
        status = manager.get_status()
        if status.state != SchemaState.CURRENT:
            raise SchemaVersionError(
                f"Schema is not current (state={status.state.value}, "
                f"current={status.current_version}, head={status.head_version}). "
                "Use ensure_schema=True to auto-migrate."
            )

    def get_schema_status(self) -> SchemaStatus:
        """Get current schema status for diagnostics."""
        manager = SchemaManager(self._lg, self._db.engine)
        return manager.get_status()

    def health_check(self) -> dict[str, Any]:
        """
        Check database connectivity.

        Returns:
            Dict with status and response time
        """
        return self._db.health_check()

    def get_stats(self) -> dict[str, Any]:
        """
        Get statistics for this context across all collections.

        Returns:
            Dict with counts for each collection type
        """
        return {
            "context_key": self._context.context_key,
            "content": self._content.count(),
            "atomic": self._atomic.get_stats(),
        }
