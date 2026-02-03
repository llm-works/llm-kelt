"""Atomic memory protocol - aggregates all atomic clients."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from llm_learn.core.embedding import EmbeddingStore

from .clients import (
    AssertionsClient,
    DirectivesClient,
    FeedbackClient,
    InteractionsClient,
    PredictionsClient,
    PreferencesClient,
    SolutionsClient,
)
from .embedding import EmbeddingAdapter

if TYPE_CHECKING:
    from llm_learn.inference.embedder import Embedder


class Protocol:
    """
    Atomic memory protocol - fact-based knowledge storage.

    Aggregates all atomic clients under a single interface. Access via LearnClient.atomic.

    Usage:
        learn = LearnClient(profile_id="a3f8b2c1...")

        # Access atomic primitives
        learn.atomic.assertions.add("User prefers Python")
        learn.atomic.solutions.record(agent_name="reviewer", ...)
        learn.atomic.predictions.record(hypothesis="X will happen", confidence=0.7)

        # Embedding operations
        learn.atomic.embeddings.embed_fact(fact, "text-embedding-3-small")
        results = learn.atomic.embeddings.search_similar(query_embedding, model_name)
    """

    def __init__(
        self,
        session_factory: Callable[[], Any],
        profile_id: str,
        *,
        embedder: Embedder | None = None,
        embedding_store: EmbeddingStore | None = None,
    ) -> None:
        """
        Initialize Atomic memory protocol.

        Args:
            session_factory: Database session factory.
            profile_id: Profile ID (32-char hash) to scope all operations to.
            embedder: Optional embedder for generating embeddings.
            embedding_store: Optional embedding store for vector operations.
        """
        self._session_factory = session_factory
        self._profile_id = profile_id
        self._embedder = embedder
        self._embedding_store = embedding_store

        # Lazy-initialized clients
        self._assertions: AssertionsClient | None = None
        self._solutions: SolutionsClient | None = None
        self._predictions: PredictionsClient | None = None
        self._feedback: FeedbackClient | None = None
        self._directives: DirectivesClient | None = None
        self._interactions: InteractionsClient | None = None
        self._preferences: PreferencesClient | None = None
        self._embedding_adapter: EmbeddingAdapter | None = None

    @property
    def assertions(self) -> AssertionsClient:
        """Simple facts about the user."""
        if self._assertions is None:
            self._assertions = AssertionsClient(self._session_factory, self._profile_id)
        return self._assertions

    @property
    def solutions(self) -> SolutionsClient:
        """Agent problem/answer records."""
        if self._solutions is None:
            self._solutions = SolutionsClient(self._session_factory, self._profile_id)
        return self._solutions

    @property
    def predictions(self) -> PredictionsClient:
        """Hypothesis tracking for calibration."""
        if self._predictions is None:
            self._predictions = PredictionsClient(self._session_factory, self._profile_id)
        return self._predictions

    @property
    def feedback(self) -> FeedbackClient:
        """Explicit user signals on content."""
        if self._feedback is None:
            self._feedback = FeedbackClient(self._session_factory, self._profile_id)
        return self._feedback

    @property
    def directives(self) -> DirectivesClient:
        """Standing user instructions."""
        if self._directives is None:
            self._directives = DirectivesClient(self._session_factory, self._profile_id)
        return self._directives

    @property
    def interactions(self) -> InteractionsClient:
        """Implicit behavioral signals."""
        if self._interactions is None:
            self._interactions = InteractionsClient(self._session_factory, self._profile_id)
        return self._interactions

    @property
    def preferences(self) -> PreferencesClient:
        """DPO training pairs."""
        if self._preferences is None:
            self._preferences = PreferencesClient(self._session_factory, self._profile_id)
        return self._preferences

    @property
    def embeddings(self) -> EmbeddingAdapter:
        """Embedding operations for atomic facts."""
        if self._embedding_adapter is None:
            if self._embedding_store is None:
                raise RuntimeError("No embedding store configured")
            self._embedding_adapter = EmbeddingAdapter(
                self._session_factory,
                self._profile_id,
                self._embedding_store,
                self._embedder,
            )
        return self._embedding_adapter

    def get_stats(self) -> dict[str, int]:
        """
        Get counts for all atomic collections.

        Returns:
            Dict with counts for each collection type.
        """
        return {
            "assertions": self.assertions.count(),
            "solutions": self.solutions.count(),
            "predictions": self.predictions.count(),
            "feedback": self.feedback.count(),
            "directives": self.directives.count(),
            "interactions": self.interactions.count(),
            "preferences": self.preferences.count(),
        }
