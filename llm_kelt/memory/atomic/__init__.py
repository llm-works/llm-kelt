"""Atomic memory model - fact-based knowledge storage.

The atomic model stores all knowledge as discrete facts with type-specific
detail tables. It's called "atomic" because each piece of knowledge is
an atomic, self-contained unit.

Usage:
    from llm_kelt.memory.atomic import Protocol

    protocol = Protocol(lg, session_factory, context_key)
    protocol.assertions.add("User prefers concise responses")
    protocol.feedback.record("positive", content_id=123)
"""

from .clients import (
    AssertionsClient,
    DirectivesClient,
    FeedbackClient,
    InteractionsClient,
    PredictionsClient,
    PreferencesClient,
    SolutionsClient,
)
from .embedding import EmbeddingAdapter, EmbeddingFilter
from .models import (
    DirectiveDetails,
    Fact,
    FactRelationship,
    FeedbackDetails,
    InteractionDetails,
    PredictionDetails,
    PreferenceDetails,
    RelType,
    SolutionDetails,
)
from .protocol import Protocol
from .relationships import RelationshipsClient

__all__ = [
    # Protocol
    "Protocol",
    # Models
    "Fact",
    "FactRelationship",
    "RelType",
    "SolutionDetails",
    "PredictionDetails",
    "FeedbackDetails",
    "DirectiveDetails",
    "InteractionDetails",
    "PreferenceDetails",
    # Clients
    "AssertionsClient",
    "RelationshipsClient",
    "SolutionsClient",
    "PredictionsClient",
    "FeedbackClient",
    "DirectivesClient",
    "InteractionsClient",
    "PreferencesClient",
    # Embedding
    "EmbeddingAdapter",
    "EmbeddingFilter",
]
