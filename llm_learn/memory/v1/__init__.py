"""Memory v1 - unified fact-based knowledge storage."""

from .clients import (
    AssertionsClient,
    DirectivesClient,
    FactClient,
    FeedbackClient,
    InteractionsClient,
    PredictionsClient,
    PreferencesClient,
    ScoredFact,
    SolutionsClient,
)
from .interface import Protocol
from .models import (
    DirectiveDetails,
    Fact,
    FactEmbedding,
    FeedbackDetails,
    InteractionDetails,
    PredictionDetails,
    PreferenceDetails,
    SolutionDetails,
)

__all__ = [
    # Interface
    "Protocol",
    # Models
    "Fact",
    "FactEmbedding",
    "SolutionDetails",
    "PredictionDetails",
    "FeedbackDetails",
    "DirectiveDetails",
    "InteractionDetails",
    "PreferenceDetails",
    # Clients
    "AssertionsClient",
    "DirectivesClient",
    "FactClient",
    "FeedbackClient",
    "InteractionsClient",
    "PredictionsClient",
    "PreferencesClient",
    "ScoredFact",
    "SolutionsClient",
]
