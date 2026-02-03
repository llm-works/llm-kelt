"""Main LearnClient - primary API entry point."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .collection.content import ContentClient
from .core.database import Database
from .memory import v1

if TYPE_CHECKING:
    from .memory.v1 import (
        AssertionsClient,
        DirectivesClient,
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

    Memory v1 uses a unified fact-based architecture where all knowledge
    is stored in memv1_facts with type-specific detail tables.

    Usage:
        from llm_learn import LearnClient

        # Create client scoped to a profile
        learn = LearnClient(profile_id=123)

        # Access v1 memory primitives
        learn.v1.assertions.add("Prefers concise explanations", category="preferences")
        learn.v1.solutions.record(agent_name="reviewer", problem="...", ...)
        learn.v1.predictions.record(hypothesis="X will happen", confidence=0.7)
        learn.v1.feedback.record(signal="positive", content_id=456)
        learn.v1.preferences.record(context="...", chosen="...", rejected="...")

        # Convenience aliases (shorthand for learn.v1.*)
        learn.assertions.add(...)
        learn.solutions.record(...)
    """

    def __init__(
        self,
        profile_id: int,
        config_path: str | None = None,
        db_key: str = "main",
        database: Database | None = None,
    ) -> None:
        """
        Initialize LearnClient scoped to a specific profile.

        Args:
            profile_id: Profile ID to scope all operations to
            config_path: Path to config file. If None, uses etc/infra.yaml
            db_key: Database configuration key (default: "main")
            database: Optional pre-configured Database instance
        """
        self._profile_id = profile_id

        if database is not None:
            self._db = database
        else:
            if config_path is None:
                config_path = "etc/infra.yaml"
            self._db = Database.from_config(config_path, db_key)

        # Content client (not part of memory v1)
        self._content = ContentClient(self._db.session, profile_id)

        # Memory v1 protocol
        self._v1 = v1.Protocol(self._db.session, profile_id)

    @property
    def profile_id(self) -> int:
        """Get the profile ID this client is scoped to."""
        return self._profile_id

    @property
    def v1(self) -> Protocol:
        """Access memory v1 protocol."""
        return self._v1

    @property
    def content(self) -> ContentClient:
        """Access content storage API."""
        return self._content

    # Convenience aliases for v1 primitives
    @property
    def assertions(self) -> AssertionsClient:
        """Shorthand for v1.assertions."""
        return self._v1.assertions

    @property
    def facts(self) -> AssertionsClient:
        """Alias for assertions."""
        return self._v1.assertions

    @property
    def solutions(self) -> SolutionsClient:
        """Shorthand for v1.solutions."""
        return self._v1.solutions

    @property
    def predictions(self) -> PredictionsClient:
        """Shorthand for v1.predictions."""
        return self._v1.predictions

    @property
    def feedback(self) -> FeedbackClient:
        """Shorthand for v1.feedback."""
        return self._v1.feedback

    @property
    def directives(self) -> DirectivesClient:
        """Shorthand for v1.directives."""
        return self._v1.directives

    @property
    def interactions(self) -> InteractionsClient:
        """Shorthand for v1.interactions."""
        return self._v1.interactions

    @property
    def preferences(self) -> PreferencesClient:
        """Shorthand for v1.preferences."""
        return self._v1.preferences

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
            "v1": self._v1.get_stats(),
        }
