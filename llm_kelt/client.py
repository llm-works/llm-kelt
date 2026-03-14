"""Main Client - primary API entry point."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from appinfra.dot_dict import DotDict
from appinfra.log import Logger
from llm_infer.client import ChatClient

from .core.content import ContentStore
from .core.database import Database
from .core.embedding import EmbeddingStore
from .core.errors import SchemaVersionError
from .core.schema import SchemaManager, SchemaState, SchemaStatus
from .inference.context import ContextBuilder
from .inference.embedder import Embedder
from .inference.query import ContextQuery
from .memory import atomic
from .memory.isolation import ClientContext

if TYPE_CHECKING:
    from pathlib import Path

    from .memory.atomic import Protocol
    from .scoped_client import ScopedClient
    from .training import Factory as TrainFactory


class Client:
    """
    Main client for the Kelt framework, scoped to an isolation context.

    Provides unified access to all framework capabilities:
    - kelt.atomic.* - Fact-based memory storage (assertions, solutions, feedback, etc.)
    - kelt.train.* - Training manifest and execution
    - kelt.query - Context-aware LLM queries

    Usage (via factory - recommended):
        from appinfra.config import Config
        from appinfra.log import LogConfig, LoggerFactory
        from llm_kelt import ClientFactory, ClientContext

        config = Config("etc/llm-kelt.yaml")
        lg = LoggerFactory.create_root(LogConfig.from_params(level="info"))

        factory = ClientFactory(lg)
        context = ClientContext(context_key="my-agent", schema_name="public")
        kelt = factory.create_from_config(context=context, config=config)

    Usage (direct - for testing or shared resources):
        from llm_kelt import Client, ClientContext

        context = ClientContext(context_key="my-agent", schema_name="public")
        kelt = Client(
            database=db, context=context, lg=lg,
            embedder=embedder, llm_client=llm_client,
        )

    Memory API:
        kelt.atomic.assertions.add("Prefers concise explanations", category="preferences")
        kelt.atomic.solutions.record(agent_name="reviewer", problem="...", ...)
        kelt.atomic.feedback.record(signal="positive", content_id=456)

    Training API:
        manifest = kelt.train.manifest.create(
            key="my-adapter",
            method="dpo",
            model="Qwen/Qwen2.5-7B-Instruct",
            data=[{"prompt": "...", "chosen": "...", "rejected": "..."}],
        )
        kelt.train.manifest.submit(manifest)
        result = kelt.train.dpo.train(manifest)

    Query API (requires llm_client):
        response = await kelt.query.ask("What's a good approach?")
    """

    def __init__(
        self,
        lg: Logger,
        database: Database,
        context: ClientContext,
        embedder: Embedder | None = None,
        llm_client: ChatClient | None = None,
        kelt_config: DotDict | None = None,
        training_config: DotDict | None = None,
        ensure_schema: bool = True,
    ) -> None:
        """
        Initialize Client with isolation context.

        Args:
            database: Database instance
            context: ClientContext for data partitioning (any string format)
            lg: Optional logger instance
            embedder: Optional embedder for generating embeddings
            llm_client: Optional LLM client for context-aware queries
            kelt_config: Optional kelt settings (config.kelt section)
            training_config: Optional training settings (config.training section)
            ensure_schema: If True (default), auto-migrate schema on init
        """
        self._db = database
        self._context = context
        self._lg = lg
        self._embedder = embedder
        self._llm_client = llm_client
        self._kelt_config = kelt_config
        self._training_config = training_config
        self._ensure_schema = ensure_schema

        self._ensure_schema_config(ensure=ensure_schema)
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
        self._train: TrainFactory | None = None

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
    def context(self) -> ClientContext:
        """
        Get isolation context for this client.

        Returns the current ClientContext with context_key and schema_name.
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
    ) -> Client:
        """
        Return new client with isolation overrides.

        Only provided fields are overridden; omitted fields keep their current values.
        You can explicitly set a field to None to clear it.

        Args:
            context_key: New context_key, or None to clear. Omit to keep current.
            schema_name: New schema_name, or None to clear. Omit to keep current.
            **kwargs: Additional fields for extensibility

        Returns:
            New Client with merged context

        Example:
            # Override just schema (keeps same context_key)
            kelt.with_isolation(schema_name="public")

            # Clear context_key to None, keep schema_name
            kelt.with_isolation(context_key=None)

            # Override both
            kelt.with_isolation(
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
        return Client(
            database=self._db,
            context=merged,
            lg=self._lg,
            embedder=self._embedder,
            llm_client=self._llm_client,
            kelt_config=self._kelt_config,
            training_config=self._training_config,
            ensure_schema=False,  # Don't re-run schema checks
        )

    def with_schema(self, schema_name: str) -> ScopedClient:
        """
        Get a client scoped to a specific schema.

        All operations on the returned client use the specified schema.
        If ensure_schema was True at client construction, the schema and
        tables are created lazily on first use.

        This is the preferred way to perform multi-schema operations from
        a single Client instance. Unlike with_isolation(), this does not
        create a new Client - it creates a lightweight ScopedClient that
        shares resources with the parent.

        Args:
            schema_name: PostgreSQL schema name

        Returns:
            ScopedClient bound to the schema

        Example:
            # Schema-agnostic client
            client = KeltClient(database=db, context_key="my-agent", ensure_schema=True)

            # Schema specified at operation time
            client.with_schema("hn_exp").atomic.solutions.record(...)
            client.with_schema("playground").atomic.facts.add(...)
        """
        from .scoped_client import ScopedClient

        return ScopedClient(
            lg=self._lg,
            parent=self,
            schema_name=schema_name,
            ensure_schema=self._ensure_schema,
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
    def train(self) -> TrainFactory:
        """Access training factory for manifest lifecycle and training execution.

        Requires adapters.lora.base_path to be configured in kelt_config.

        Raises:
            RuntimeError: If adapters.lora.base_path is not configured.
        """
        if self._train is None:
            from .training.factory import Factory

            registry_path = self._get_registry_path()
            if registry_path is None:
                raise RuntimeError(
                    "Training not configured: adapters.lora.base_path not set in kelt_config"
                )
            self._train = Factory(self._lg, registry_path, self._get_default_profiles())
        return self._train

    def _get_registry_path(self) -> Path | None:
        """Get adapter registry path from kelt_config."""
        if self._kelt_config is None:
            return None
        adapters = getattr(self._kelt_config, "adapters", None)
        if adapters is None:
            return None
        lora = getattr(adapters, "lora", None)
        if lora is None:
            return None
        base_path = getattr(lora, "base_path", None)
        if base_path is None:
            return None
        from pathlib import Path

        return Path(base_path)

    def _get_default_profiles(self) -> dict[str, dict]:
        """Get default training profiles from training_config."""
        if self._training_config is None:
            return {}
        default_profiles = getattr(self._training_config, "default_profiles", None)
        if default_profiles is None:
            return {}
        # Convert DotDict to plain dict
        return {k: dict(v) for k, v in default_profiles.items()}

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
    def kelt_config(self) -> DotDict | None:
        """Access kelt configuration (memory, embedding, default_system_prompt, etc.)."""
        return self._kelt_config

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
            raise RuntimeError("LLM client not configured. Pass llm_client to Client.")
        return self._context_query

    def _get_default_system_prompt(self) -> str:
        """Get default system prompt from kelt config."""
        if self._kelt_config is None:
            return ""
        return getattr(self._kelt_config, "default_system_prompt", "") or ""

    def _ensure_schema_config(self, *, ensure: bool) -> None:
        """Configure Database schema from ClientContext if needed.

        When ClientContext.schema_name is set but Database wasn't configured with
        a schema, this dynamically configures PG with the schema from context.

        For schema-agnostic clients (context.schema_name is None), this is a no-op.
        Schema selection happens at operation time via with_schema().

        Args:
            ensure: If True, configure PG with schema from context.
        """
        context_schema = self._context.schema_name
        db_schema = self._db.schema

        # Schema-agnostic or already configured - nothing to do
        if context_schema is None or context_schema == db_schema:
            return

        # Schema conflict: context and DB have different schemas
        if db_schema is not None and context_schema != db_schema:
            raise ValueError(
                f"Schema conflict: context specifies '{context_schema}' but "
                f"database is configured for '{db_schema}'. Use with_schema() "
                "for per-operation schema selection instead."
            )

        # Context has a schema but DB doesn't - configure it (backward compat)
        if db_schema is None and ensure:
            self._db.configure_schema(context_schema)
            self._lg.info(
                "configured database schema from context",
                extra={"schema": context_schema},
            )

    def _verify_schema(self, *, ensure: bool) -> None:
        """Verify database schema is current, optionally auto-migrating.

        Args:
            ensure: If True, create database and run migrations automatically.
                    If False, only verify — raise SchemaVersionError if not current.
        """
        schema_name = self._context.schema_name or self._db.schema
        if ensure:
            self._db.ensure_database()
            self._db.ensure_pg_schema()  # Create PostgreSQL schema if configured
            manager = SchemaManager(self._lg, self._db.engine, schema_name=schema_name)
            manager.ensure_schema()
            return

        manager = SchemaManager(self._lg, self._db.engine, schema_name=schema_name)
        status = manager.get_status()
        if status.state != SchemaState.CURRENT:
            raise SchemaVersionError(
                f"Schema is not current (state={status.state.value}, "
                f"current={status.current_version}, head={status.head_version}). "
                "Use ensure_schema=True to auto-migrate."
            )

    def get_schema_status(self) -> SchemaStatus:
        """Get current schema status for diagnostics."""
        schema_name = self._context.schema_name or self._db.schema
        manager = SchemaManager(self._lg, self._db.engine, schema_name=schema_name)
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
