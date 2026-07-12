"""Atomic memory protocol - aggregates all atomic clients."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from appinfra.log import Logger
from sqlalchemy.exc import OperationalError, ProgrammingError

from llm_kelt.embedding import Factory as EmbeddingFactory
from llm_kelt.embedding.types import QuantizationFormat

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
from .relationships import RelationshipsClient

if TYPE_CHECKING:
    from llm_infer.client import EmbeddingClient


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
        results = kelt.atomic.embeddings.search_similar(query_embedding, model)
    """

    def __init__(
        self,
        lg: Logger,
        session_factory: Callable[[], Any],
        context_key: str | None,
        *,
        embedder: EmbeddingClient | None = None,
        embedding_factory: EmbeddingFactory | None = None,
        embedding_format: QuantizationFormat = QuantizationFormat.F16,
        embedding_dimensions: int | None = None,
        embedding_schema: str | None = None,
    ) -> None:
        """
        Initialize Atomic memory protocol.

        Args:
            lg: Logger instance for all atomic operations.
            session_factory: Database session factory.
            context_key: Context key to scope all operations to (None = no filtering).
            embedder: Optional embedder for generating embeddings.
            embedding_factory: Factory for creating dimension-specific embedding stores.
            embedding_format: Quantization format for embeddings (default: F16).
            embedding_dimensions: Default output dimensions for embeddings.
            embedding_schema: Postgres schema for embedding tables. Forwarded to
                the EmbeddingAdapter so every dimension-specific store binds a
                schema-qualified ORM model.
        """
        self._lg = lg
        self._session_factory = session_factory
        self._context_key = context_key
        self._embedder = embedder
        self._embedding_factory = embedding_factory
        self._embedding_format = embedding_format
        self._embedding_dimensions = embedding_dimensions
        self._embedding_schema = embedding_schema

        # Lazy-initialized clients
        self._assertions: AssertionsClient | None = None
        self._solutions: SolutionsClient | None = None
        self._predictions: PredictionsClient | None = None
        self._feedback: FeedbackClient | None = None
        self._directives: DirectivesClient | None = None
        self._interactions: InteractionsClient | None = None
        self._preferences: PreferencesClient | None = None
        self._relationships: RelationshipsClient | None = None

        self._embedding_adapter: EmbeddingAdapter | None = self._build_adapter()

    def _build_adapter(self) -> EmbeddingAdapter | None:
        """Eagerly construct the embedding adapter when a factory is available."""
        if self._embedding_factory is None:
            return None
        return EmbeddingAdapter(
            session_factory=self._session_factory,
            context_key=self._context_key,
            factory=self._embedding_factory,
            format=self._embedding_format,
            embedder=self._embedder,
            default_dimensions=self._embedding_dimensions,
            schema=self._embedding_schema,
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
    def relationships(self) -> RelationshipsClient:
        """Fact relationship edges (contradicts, supports, etc.)."""
        if self._relationships is None:
            self._relationships = RelationshipsClient(
                self._lg, self._session_factory, self._context_key
            )
        return self._relationships

    @property
    def embeddings(self) -> EmbeddingAdapter:
        """Embedding operations for atomic facts."""
        if self._embedding_adapter is None:
            self._embedding_adapter = self._build_adapter()
            if self._embedding_adapter is None:
                raise RuntimeError("No embedding factory configured")
        return self._embedding_adapter

    def get_stats(self) -> dict[str, int]:
        """
        Get counts for all atomic collections.

        Returns:
            Dict with counts for each collection type.
        """
        stats: dict[str, int] = {
            "assertions": self.assertions.count(),
            "solutions": self.solutions.count(),
            "predictions": self.predictions.count(),
            "feedback": self.feedback.count(),
            "directives": self.directives.count(),
            "interactions": self.interactions.count(),
            "preferences": self.preferences.count(),
        }
        try:
            stats["relationships"] = self.relationships.count()
        except (ProgrammingError, OperationalError) as e:
            self._lg.warning("failed to count relationships", extra={"exception": e})
            stats["relationships"] = 0
        return stats
