"""Atomic memory protocol - aggregates all atomic clients."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from appinfra.log import Logger

from llm_kelt.core.embedding import EmbeddingStore

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
    from llm_kelt.inference.embedder import Embedder


class Protocol:
    """
    Atomic memory protocol - fact-based knowledge storage.

    Aggregates all atomic clients under a single interface. Access via Client.atomic.

    Usage:
        from llm_kelt import Client, ClientContext
        context = ClientContext(context_key="my-agent")
        kelt = Client(database=db, context=context)

        # Access atomic primitives
        kelt.atomic.assertions.add("User prefers Python")
        kelt.atomic.solutions.record(agent_name="reviewer", ...)
        kelt.atomic.predictions.record(hypothesis="X will happen", confidence=0.7)

        # Embedding operations
        kelt.atomic.embeddings.embed_fact(fact, "text-embedding-3-small")
        results = kelt.atomic.embeddings.search_similar(query_embedding, model_name)
    """

    def __init__(
        self,
        lg: Logger,
        session_factory: Callable[[], Any],
        context_key: str | None,
        *,
        embedder: Embedder | None = None,
        embedding_store: EmbeddingStore | None = None,
    ) -> None:
        """
        Initialize Atomic memory protocol.

        Args:
            lg: Logger instance for all atomic operations.
            session_factory: Database session factory.
            context_key: Context key to scope all operations to (None = no filtering).
            embedder: Optional embedder for generating embeddings.
            embedding_store: Optional embedding store for vector operations.
        """
        self._lg = lg
        self._session_factory = session_factory
        self._context_key = context_key
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

        # Eagerly initialize embedding adapter so clients can use it
        self._embedding_adapter: EmbeddingAdapter | None = None
        if self._embedding_store is not None:
            self._embedding_adapter = EmbeddingAdapter(
                self._session_factory,
                self._context_key,
                self._embedding_store,
                self._embedder,
            )

    @property
    def assertions(self) -> AssertionsClient:
        """Simple facts about the user."""
        if self._assertions is None:
            self._assertions = AssertionsClient(
                self._lg, self._session_factory, self._context_key, self._embedding_adapter
            )
        return self._assertions

    @property
    def solutions(self) -> SolutionsClient:
        """Agent problem/answer records."""
        if self._solutions is None:
            self._solutions = SolutionsClient(
                self._lg, self._session_factory, self._context_key, self._embedding_adapter
            )
        return self._solutions

    @property
    def predictions(self) -> PredictionsClient:
        """Hypothesis tracking for calibration."""
        if self._predictions is None:
            self._predictions = PredictionsClient(
                self._lg, self._session_factory, self._context_key, self._embedding_adapter
            )
        return self._predictions

    @property
    def feedback(self) -> FeedbackClient:
        """Explicit user signals on content."""
        if self._feedback is None:
            self._feedback = FeedbackClient(self._lg, self._session_factory, self._context_key)
        return self._feedback

    @property
    def directives(self) -> DirectivesClient:
        """Standing user instructions."""
        if self._directives is None:
            self._directives = DirectivesClient(
                self._lg, self._session_factory, self._context_key, self._embedding_adapter
            )
        return self._directives

    @property
    def interactions(self) -> InteractionsClient:
        """Implicit behavioral signals."""
        if self._interactions is None:
            self._interactions = InteractionsClient(
                self._lg, self._session_factory, self._context_key
            )
        return self._interactions

    @property
    def preferences(self) -> PreferencesClient:
        """DPO training pairs."""
        if self._preferences is None:
            self._preferences = PreferencesClient(
                self._lg, self._session_factory, self._context_key
            )
        return self._preferences

    @property
    def embeddings(self) -> EmbeddingAdapter:
        """Embedding operations for atomic facts."""
        # Defensive: adapter is eagerly created in __init__ if embedding_store is set
        # This lazy fallback handles edge cases where adapter was reset or not initialized
        if self._embedding_adapter is None:
            if self._embedding_store is None:
                raise RuntimeError("No embedding store configured")
            self._embedding_adapter = EmbeddingAdapter(
                self._session_factory,
                self._context_key,
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
