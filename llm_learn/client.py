"""Main LearnClient - primary API entry point."""

from typing import Any

from .collection.content import ContentClient
from .collection.directives import DirectivesClient
from .collection.facts import FactsClient
from .collection.feedback import FeedbackClient
from .collection.interactions import InteractionsClient
from .collection.predictions import PredictionsClient
from .collection.preferences import PreferencesClient
from .core.database import Database


class LearnClient:
    """
    Main client for the Learn framework, scoped to a profile.

    Provides unified access to all data collection APIs for a specific profile.
    All operations are automatically filtered by profile_id.

    Usage:
        from llm_learn import LearnClient

        # Create client scoped to a profile
        learn = LearnClient(profile_id=123)

        # Record feedback for this profile
        learn.feedback.record(
            content_text="Full article...",
            signal="positive",
        )

        # Add facts about the user
        learn.facts.add(
            "Prefers concise explanations",
            category="preferences",
        )

        # Record preference pair
        learn.preferences.record(
            context="Summarize this",
            chosen="Concise version",
            rejected="Verbose version",
        )

        # Record prediction
        pred_id = learn.predictions.record(
            hypothesis="X will happen",
            confidence=0.7,
        )
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

        # Initialize sub-clients with session factory and profile_id
        self._content = ContentClient(self._db.session, profile_id)
        self._facts = FactsClient(self._db.session, profile_id)
        self._feedback = FeedbackClient(self._db.session, profile_id)
        self._preferences = PreferencesClient(self._db.session, profile_id)
        self._interactions = InteractionsClient(self._db.session, profile_id)
        self._predictions = PredictionsClient(self._db.session, profile_id)
        self._directives = DirectivesClient(self._db.session, profile_id)

    @property
    def profile_id(self) -> int:
        """Get the profile ID this client is scoped to."""
        return self._profile_id

    @property
    def content(self) -> ContentClient:
        """Access content storage API."""
        return self._content

    @property
    def facts(self) -> FactsClient:
        """Access facts collection API."""
        return self._facts

    @property
    def feedback(self) -> FeedbackClient:
        """Access feedback collection API."""
        return self._feedback

    @property
    def preferences(self) -> PreferencesClient:
        """Access preferences collection API."""
        return self._preferences

    @property
    def interactions(self) -> InteractionsClient:
        """Access interactions collection API."""
        return self._interactions

    @property
    def predictions(self) -> PredictionsClient:
        """Access predictions collection API."""
        return self._predictions

    @property
    def directives(self) -> DirectivesClient:
        """Access directives collection API."""
        return self._directives

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
            "facts": self._facts.count(),
            "feedback": self._feedback.count(),
            "preferences": self._preferences.count(),
            "interactions": self._interactions.count(),
            "predictions": self._predictions.count(),
            "directives": self._directives.count(),
        }
