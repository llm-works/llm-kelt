"""Main LearnClient - primary API entry point."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from appinfra.dot_dict import DotDict
from appinfra.log import Logger
from llm_infer.client import LLMClient
from sqlalchemy.exc import IntegrityError

from .core.content import ContentStore
from .core.database import Database
from .core.domain import Domain
from .core.embedding import EmbeddingStore
from .core.exceptions import SchemaVersionError, ValidationError
from .core.identity import ProfileIdentity
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
        identity: ProfileIdentity | None = None,
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
            identity: Optional ProfileIdentity for hierarchy creation. If not provided,
                     creates legacy "default" workspace with profile_id as slug.
        """
        self._lg = lg
        self._profile_id = profile_id
        self._identity = identity
        self._db = database
        self._embedder = embedder
        self._llm_client = llm_client
        self._learn_config = learn_config

        # Validate profile_id is proper hash
        if not isinstance(profile_id, str) or len(profile_id) != 32:
            raise ValidationError(
                f"profile_id must be 32-char hex hash, got: {profile_id!r}. "
                "Use IdentityResolver to generate proper IDs."
            )

        self._verify_schema(ensure=ensure_schema)
        self._ensure_profile()
        self._setup_stores()
        self._setup_query_interface()

    @classmethod
    def from_identity(
        cls,
        lg: Logger,
        identity: ProfileIdentity,
        database: Database,
        embedder: Embedder | None = None,
        llm_client: LLMClient | None = None,
        learn_config: DotDict | None = None,
        ensure_schema: bool = True,
    ) -> LearnClient:
        """Create LearnClient from ProfileIdentity (recommended).

        This is the preferred way to create a LearnClient when you want to specify
        the full domain/workspace/profile hierarchy.

        Args:
            lg: Logger instance
            identity: Resolved ProfileIdentity with all IDs determined
            database: Database instance
            embedder: Optional embedder for generating embeddings
            llm_client: Optional LLM client for context-aware queries
            learn_config: Optional learn settings
            ensure_schema: If True, auto-migrate schema on init

        Returns:
            LearnClient instance scoped to the profile

        Example:
            identity = IdentityResolver.resolve({
                "domain": "acme",
                "workspace": "production",
                "name": "code-reviewer"
            })
            client = LearnClient.from_identity(lg, identity, database)
        """
        return cls(
            lg=lg,
            profile_id=identity.profile_id,
            database=database,
            embedder=embedder,
            llm_client=llm_client,
            learn_config=learn_config,
            ensure_schema=ensure_schema,
            identity=identity,
        )

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
        """Ensure the profile row and its parent hierarchy exist.

        Creates the full domain → workspace → profile hierarchy if identity is provided.
        Otherwise creates legacy "default" workspace for backwards compatibility.

        Uses try/except IntegrityError to handle concurrent callers safely.
        """
        # Fast path: profile already exists
        with self._db.session() as session:
            if session.get(Profile, self._profile_id):
                return

        if self._identity is not None:
            # New path: create full hierarchy from ProfileIdentity
            self._ensure_hierarchy_from_identity()
        else:
            # Legacy path: create default workspace + profile
            self._ensure_legacy_profile()

    def _ensure_hierarchy_from_identity(self) -> None:
        """Create full domain → workspace → profile hierarchy from ProfileIdentity."""
        assert self._identity is not None

        # Level 1: Ensure domain exists (if specified)
        if self._identity.domain is not None:
            assert self._identity.domain_id is not None
            self._ensure_domain_exists()

        # Level 2: Ensure workspace exists
        self._ensure_workspace_exists()

        # Level 3: Ensure profile exists
        self._ensure_profile_exists()

    def _ensure_domain_exists(self) -> None:
        """Ensure domain exists in database."""
        assert (
            self._identity is not None
            and self._identity.domain is not None
            and self._identity.domain_id is not None
        )
        try:
            with self._db.session() as session:
                if not session.get(Domain, self._identity.domain_id):
                    session.add(
                        Domain(
                            id=self._identity.domain_id,
                            slug=self._identity.domain,
                            name=self._identity.domain.title(),
                        )
                    )
        except IntegrityError:
            self._lg.debug("domain already created by concurrent process")

    def _ensure_workspace_exists(self) -> None:
        """Ensure workspace exists in database."""
        assert self._identity is not None
        try:
            with self._db.session() as session:
                if not session.get(Workspace, self._identity.workspace_id):
                    session.add(
                        Workspace(
                            id=self._identity.workspace_id,
                            domain_id=self._identity.domain_id,
                            slug=self._identity.workspace,
                            name=self._identity.workspace.title(),
                        )
                    )
        except IntegrityError:
            self._lg.debug("workspace already created by concurrent process")

    def _ensure_profile_exists(self) -> None:
        """Ensure profile exists in database."""
        assert self._identity is not None
        try:
            with self._db.session() as session:
                session.add(
                    Profile(
                        id=self._identity.profile_id,
                        workspace_id=self._identity.workspace_id,
                        slug=self._identity.name,
                        name=self._identity.name.title(),
                        active=True,
                    )
                )
        except IntegrityError:
            self._lg.debug("profile already created by concurrent process")

    def _ensure_legacy_profile(self) -> None:
        """Create legacy default workspace + profile (backwards compatibility)."""
        workspace_id = Workspace.generate_id(None, "default")

        # Ensure workspace exists
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
