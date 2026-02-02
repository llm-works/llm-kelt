"""Memory v1 interface - aggregates all v1 clients."""

from collections.abc import Callable
from typing import Any

from .clients import (
    AssertionsClient,
    DirectivesClient,
    FeedbackClient,
    InteractionsClient,
    PredictionsClient,
    PreferencesClient,
    SolutionsClient,
)


class Protocol:
    """
    Memory v1 protocol - fact-based knowledge storage.

    Aggregates all v1 clients under a single interface. Access via LearnClient.v1.

    Usage:
        learn = LearnClient(profile_id=123)

        # Access v1 primitives
        learn.v1.assertions.add("User prefers Python")
        learn.v1.solutions.record(agent_name="reviewer", ...)
        learn.v1.predictions.record(hypothesis="X will happen", confidence=0.7)
    """

    def __init__(self, session_factory: Callable[[], Any], profile_id: int) -> None:
        """
        Initialize Memory v1 interface.

        Args:
            session_factory: Database session factory
            profile_id: Profile ID to scope all operations to
        """
        self._session_factory = session_factory
        self._profile_id = profile_id

        # Initialize all v1 clients
        self._assertions = AssertionsClient(session_factory, profile_id)
        self._solutions = SolutionsClient(session_factory, profile_id)
        self._predictions = PredictionsClient(session_factory, profile_id)
        self._feedback = FeedbackClient(session_factory, profile_id)
        self._directives = DirectivesClient(session_factory, profile_id)
        self._interactions = InteractionsClient(session_factory, profile_id)
        self._preferences = PreferencesClient(session_factory, profile_id)

    @property
    def assertions(self) -> AssertionsClient:
        """Simple facts about the user."""
        return self._assertions

    @property
    def solutions(self) -> SolutionsClient:
        """Agent problem/answer records."""
        return self._solutions

    @property
    def predictions(self) -> PredictionsClient:
        """Hypothesis tracking for calibration."""
        return self._predictions

    @property
    def feedback(self) -> FeedbackClient:
        """Explicit user signals on content."""
        return self._feedback

    @property
    def directives(self) -> DirectivesClient:
        """Standing user instructions."""
        return self._directives

    @property
    def interactions(self) -> InteractionsClient:
        """Implicit behavioral signals."""
        return self._interactions

    @property
    def preferences(self) -> PreferencesClient:
        """DPO training pairs."""
        return self._preferences

    def get_stats(self) -> dict[str, int]:
        """
        Get counts for all v1 collections.

        Returns:
            Dict with counts for each collection type
        """
        return {
            "assertions": self._assertions.count(),
            "solutions": self._solutions.count(),
            "predictions": self._predictions.count(),
            "feedback": self._feedback.count(),
            "directives": self._directives.count(),
            "interactions": self._interactions.count(),
            "preferences": self._preferences.count(),
        }
