"""Preferences client for DPO training pairs."""

from typing import cast

from appinfra.db.utils import detach, detach_all
from sqlalchemy import select

from llm_learn.core.exceptions import ValidationError

from ..models import Fact, PreferenceDetails
from .base import FactClient


class PreferencesClient(FactClient[PreferenceDetails]):
    """
    Client for recording preference pairs for DPO training.

    Preference pairs capture "chosen vs rejected" responses for a given
    context, which can be used for Direct Preference Optimization training.

    Usage:
        preferences = PreferencesClient(session_factory, profile_id="a3f8b2c1...")

        # Record a preference pair
        fact_id = preferences.record(
            context="Summarize this article",
            chosen="A concise 3-sentence summary...",
            rejected="A verbose 3-paragraph summary...",
            margin=0.8,
            category="summarization",
        )

        # List by category
        pairs = preferences.list_by_category("summarization")
    """

    fact_type = "preference"
    details_model = PreferenceDetails
    details_relationship = "preference_details"

    def _validate_preference_inputs(
        self, context: str, chosen: str, rejected: str, margin: float | None
    ) -> None:
        """Validate preference pair inputs."""
        if not context or not context.strip():
            raise ValidationError("context cannot be empty")
        if not chosen or not chosen.strip():
            raise ValidationError("chosen cannot be empty")
        if not rejected or not rejected.strip():
            raise ValidationError("rejected cannot be empty")
        if margin is not None and (margin < 0.0 or margin > 1.0):
            raise ValidationError(f"margin must be between 0.0 and 1.0, got {margin}")

    def record(
        self,
        context: str,
        chosen: str,
        rejected: str,
        margin: float | None = None,
        category: str | None = None,
        metadata: dict | None = None,
    ) -> int:
        """Record a preference pair."""
        self._validate_preference_inputs(context, chosen, rejected, margin)

        with self._session_factory() as session:
            preview = context[:100] + "..." if len(context) > 100 else context
            fact = Fact(
                context_key=self.context_key,
                type=self.fact_type,
                content=f"Preference: {preview}",
                category=category,
                source="user",
                confidence=1.0,
                active=True,
            )
            session.add(fact)
            session.flush()

            details = PreferenceDetails(
                fact_id=fact.id,
                context=context.strip(),
                chosen=chosen.strip(),
                rejected=rejected.strip(),
                margin=margin,
                metadata_=metadata,
            )
            session.add(details)
            return fact.id

    def list_by_category(
        self,
        category: str,
        limit: int = 100,
        active_only: bool = True,
    ) -> list[Fact]:
        """List preference pairs in a specific category."""
        with self._session_factory() as session:
            stmt = (
                select(Fact)
                .join(PreferenceDetails)
                .where(
                    Fact.context_key == self.context_key,
                    Fact.type == self.fact_type,
                    Fact.category == category,
                )
            )

            if active_only:
                stmt = stmt.where(Fact.active == True)  # noqa: E712

            stmt = stmt.order_by(Fact.created_at.desc()).limit(limit)

            facts = list(session.scalars(stmt).all())
            for fact in facts:
                details = fact.preference_details
                if details is not None:
                    detach(details, session)
            return cast(list[Fact], detach_all(facts, session))

    def get_categories(self) -> list[str]:
        """Get list of unique categories."""
        with self._session_factory() as session:
            stmt = (
                select(Fact.category)
                .where(
                    Fact.context_key == self.context_key,
                    Fact.type == self.fact_type,
                    Fact.category.isnot(None),
                )
                .distinct()
                .order_by(Fact.category)
            )
            return [c for c in session.scalars(stmt).all() if c is not None]

    def search(
        self,
        query: str,
        limit: int = 50,
        active_only: bool = True,
    ) -> list[Fact]:
        """Search preference pairs by context."""
        with self._session_factory() as session:
            stmt = (
                select(Fact)
                .join(PreferenceDetails)
                .where(
                    Fact.context_key == self.context_key,
                    Fact.type == self.fact_type,
                    PreferenceDetails.context.ilike(f"%{query}%"),
                )
            )

            if active_only:
                stmt = stmt.where(Fact.active == True)  # noqa: E712

            stmt = stmt.order_by(Fact.created_at.desc()).limit(limit)

            facts = list(session.scalars(stmt).all())
            for fact in facts:
                details = fact.preference_details
                if details is not None:
                    detach(details, session)
            return cast(list[Fact], detach_all(facts, session))
