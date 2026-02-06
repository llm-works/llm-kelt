"""Main LearnClient - primary API entry point."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from appinfra.dot_dict import DotDict
from appinfra.log import Logger
from llm_infer.client import LLMClient
from sqlalchemy.exc import IntegrityError

from .core.content import ContentStore
from .core.database import Database
from .core.embedding import EmbeddingStore
from .core.exceptions import SchemaVersionError
from .core.profile import Profile
from .core.schema import SchemaManager, SchemaState, SchemaStatus
from .core.workspace import Workspace
from .inference.context import ContextBuilder
from .inference.embedder import Embedder
from .inference.query import ContextQuery
from .memory import atomic

if TYPE_CHECKING:
    from .memory.atomic import (
        AssertionsClient,
        DirectivesClient,
        EmbeddingAdapter,
        FeedbackClient,
        InteractionsClient,
        PredictionsClient,
        PreferencesClient,
        Protocol,
        SolutionsClient,
    )


class LearnClient:
    """
    Main client for the Learn framework, scoped to a profile.

    Provides unified access to all framework capabilities:
    - Memory storage (facts, feedback, solutions, preferences, etc.)
    - Embeddings for semantic search
    - Context-aware LLM queries

    Usage (via factory - recommended):
        from appinfra.config import Config
        from appinfra.log import LogConfig, LoggerFactory
        from llm_learn import LearnClientFactory

        config = Config("etc/llm-learn.yaml")
        lg = LoggerFactory.create_root(LogConfig.from_params(level="info"))

        factory = LearnClientFactory(lg)
        learn = factory.create_from_config(profile_id="a3f8...", config=config)

    Usage (direct - for testing or shared resources):
        from llm_learn import LearnClient

        learn = LearnClient(
            lg, profile_id="a3f8...", database=db,
            embedder=embedder, llm_client=llm_client,
        )

    Memory API:
        learn.assertions.add("Prefers concise explanations", category="preferences")
        learn.solutions.record(agent_name="reviewer", problem="...", ...)
        learn.feedback.record(signal="positive", content_id=456)

    Query API (requires llm_client):
        response = await learn.query.ask("What's a good approach?")
    """

    def __init__(
        self,
        lg: Logger,
        profile_id: str,
        database: Database,
        embedder: Embedder | None = None,
        llm_client: LLMClient | None = None,
        learn_config: DotDict | None = None,
        ensure_schema: bool = True,
    ) -> None:
        """
        Initialize LearnClient scoped to a specific profile.

        Args:
            lg: Logger instance
            profile_id: Profile ID (32-char hash) to scope all operations to
            database: Database instance
            embedder: Optional embedder for generating embeddings
            llm_client: Optional LLM client for context-aware queries
            learn_config: Optional learn settings (config.learn section).
                         Contains memory, embedding, default_system_prompt, etc.
            ensure_schema: If True (default), auto-migrate schema on init.
                          If False, verify schema is current and raise SchemaVersionError if not.
        """
        self._lg = lg
        self._profile_id = profile_id
        self._db = database
        self._embedder = embedder
        self._llm_client = llm_client
        self._learn_config = learn_config

        self._verify_schema(ensure=ensure_schema)
        self._ensure_profile()
        self._setup_stores()
        self._setup_query_interface()

    def _setup_stores(self) -> None:
        """Initialize storage components."""
        self._embedding_store = EmbeddingStore(self._db.session)
        self._content = ContentStore(self._db.session, self._profile_id)
        self._atomic = atomic.Protocol(
            self._db.session,
            self._profile_id,
            embedder=self._embedder,
            embedding_store=self._embedding_store,
        )
        self._context_builder = ContextBuilder(self._atomic.assertions)

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
    def profile_id(self) -> str:
        """Get the profile ID this client is scoped to."""
        return self._profile_id

    @property
    def atomic(self) -> Protocol:
        """Access atomic memory protocol."""
        return self._atomic

    @property
    def content(self) -> ContentStore:
        """Access content storage API."""
        return self._content

    # Convenience aliases for atomic primitives
    @property
    def assertions(self) -> AssertionsClient:
        """Shorthand for atomic.assertions."""
        return self._atomic.assertions

    @property
    def facts(self) -> AssertionsClient:
        """Alias for assertions."""
        return self._atomic.assertions

    @property
    def solutions(self) -> SolutionsClient:
        """Shorthand for atomic.solutions."""
        return self._atomic.solutions

    @property
    def predictions(self) -> PredictionsClient:
        """Shorthand for atomic.predictions."""
        return self._atomic.predictions

    @property
    def feedback(self) -> FeedbackClient:
        """Shorthand for atomic.feedback."""
        return self._atomic.feedback

    @property
    def directives(self) -> DirectivesClient:
        """Shorthand for atomic.directives."""
        return self._atomic.directives

    @property
    def interactions(self) -> InteractionsClient:
        """Shorthand for atomic.interactions."""
        return self._atomic.interactions

    @property
    def preferences(self) -> PreferencesClient:
        """Shorthand for atomic.preferences."""
        return self._atomic.preferences

    @property
    def embeddings(self) -> EmbeddingAdapter:
        """Shorthand for atomic.embeddings."""
        return self._atomic.embeddings

    @property
    def database(self) -> Database:
        """Access underlying database."""
        return self._db

    @property
    def llm_client(self) -> LLMClient | None:
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

    def _ensure_profile(self) -> None:
        """Ensure the profile row and its parent workspace exist.

        Creates a default workspace (no domain) and the profile row so that
        FK constraints on atomic_facts etc. are satisfied immediately.

        Uses try/except IntegrityError to handle concurrent callers safely.
        """
        with self._db.session() as session:
            if session.get(Profile, self._profile_id):
                return

        workspace_id = Workspace.generate_id(None, "default")

        # Ensure workspace exists (separate transaction for safe concurrency)
        try:
            with self._db.session() as session:
                if not session.get(Workspace, workspace_id):
                    session.add(Workspace(id=workspace_id, slug="default", name="Default"))
        except IntegrityError:
            self._lg.info("workspace already created by concurrent process")

        # Create profile (workspace now guaranteed to exist)
        try:
            with self._db.session() as session:
                session.add(
                    Profile(
                        id=self._profile_id,
                        workspace_id=workspace_id,
                        slug=self._profile_id,
                        name="Default",
                        active=True,
                    )
                )
        except IntegrityError:
            self._lg.info("profile already created by concurrent process")

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
        Get statistics for this profile across all collections.

        Returns:
            Dict with counts for each collection type
        """
        return {
            "profile_id": self._profile_id,
            "content": self._content.count(),
            "atomic": self._atomic.get_stats(),
        }
