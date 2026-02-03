"""Main LearnClient - primary API entry point."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .core.content import ContentStore
from .core.database import Database
from .core.embedding import EmbeddingStore
from .memory import atomic

if TYPE_CHECKING:
    from .inference.embedder import Embedder
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

    Provides unified access to all memory APIs for a specific profile.
    All operations are automatically filtered by profile_id.

    The atomic memory model uses a fact-based architecture where all knowledge
    is stored in atomic_facts with type-specific detail tables.

    Usage:
        from llm_learn import LearnClient

        # Create client scoped to a profile (32-char hash ID)
        learn = LearnClient(profile_id="a3f8b2c1d4e5f6a7b8c9d0e1f2a3b4c5")

        # Access atomic memory primitives
        learn.atomic.assertions.add("Prefers concise explanations", category="preferences")
        learn.atomic.solutions.record(agent_name="reviewer", problem="...", ...)
        learn.atomic.predictions.record(hypothesis="X will happen", confidence=0.7)
        learn.atomic.feedback.record(signal="positive", content_id=456)
        learn.atomic.preferences.record(context="...", chosen="...", rejected="...")

        # Convenience aliases (shorthand for learn.atomic.*)
        learn.assertions.add(...)
        learn.solutions.record(...)
    """

    def __init__(
        self,
        profile_id: str,
        config_path: str | None = None,
        db_key: str = "main",
        database: Database | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        """
        Initialize LearnClient scoped to a specific profile.

        Args:
            profile_id: Profile ID (32-char hash) to scope all operations to
            config_path: Path to config file. If None, uses etc/infra.yaml
            db_key: Database configuration key (default: "main")
            database: Optional pre-configured Database instance
            embedder: Optional embedder for generating embeddings
        """
        self._profile_id = profile_id

        if database is not None:
            self._db = database
        else:
            if config_path is None:
                config_path = "etc/infra.yaml"
            self._db = Database.from_config(config_path, db_key)

        # Core embedding store (always available)
        self._embedding_store = EmbeddingStore(self._db.session)

        # Content store
        self._content = ContentStore(self._db.session, profile_id)

        # Atomic memory protocol with embedding support
        self._atomic = atomic.Protocol(
            self._db.session,
            profile_id,
            embedder=embedder,
            embedding_store=self._embedding_store,
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

    def migrate(self) -> None:
        """Run database migrations to create all tables."""
        self._db.migrate()

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
