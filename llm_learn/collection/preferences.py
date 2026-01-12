"""Preference pairs collection client."""

from datetime import datetime
from typing import cast

from appinfra.db.utils import detach_all
from sqlalchemy import select

from ..core.exceptions import ValidationError
from ..core.models import PreferencePair
from .base import ProfileScopedClient


class PreferencesClient(ProfileScopedClient[PreferencePair]):
    """
    Client for recording preference pairs (DPO training data) scoped to a profile.

    Preference pairs capture user preferences between two responses
    for a given context, used for Direct Preference Optimization training.

    Usage:
        preferences = PreferencesClient(session_factory, profile_id=123)
        preferences.record(
            context="Summarize this article about AI safety",
            chosen="Concise 3-bullet summary...",
            rejected="Verbose 500-word essay...",
            margin=0.7,
            domain="synthesis",
        )
    """

    model = PreferencePair

    def record(
        self,
        context: str,
        chosen: str,
        rejected: str,
        margin: float | None = None,
        domain: str | None = None,
        metadata: dict | None = None,
    ) -> int:
        """
        Record a preference pair.

        Args:
            context: The prompt/situation
            chosen: Preferred response
            rejected: Non-preferred response
            margin: How much better chosen is (0.0-1.0)
            domain: Topic area for categorization
            metadata: Additional metadata

        Returns:
            Created preference pair ID

        Raises:
            ValidationError: If inputs are invalid
        """
        # Validate inputs
        if not context or not context.strip():
            raise ValidationError("Context cannot be empty")

        if not chosen or not chosen.strip():
            raise ValidationError("Chosen response cannot be empty")

        if not rejected or not rejected.strip():
            raise ValidationError("Rejected response cannot be empty")

        if margin is not None and (margin < 0.0 or margin > 1.0):
            raise ValidationError(f"Margin must be between 0.0 and 1.0, got {margin}")

        with self._session_factory() as session:
            pair = PreferencePair(
                profile_id=self.profile_id,
                context=context.strip(),
                chosen=chosen.strip(),
                rejected=rejected.strip(),
                margin=margin,
                domain=domain.strip() if domain else None,
                metadata_=metadata,
            )
            session.add(pair)
            session.flush()

            return pair.id

    def list_by_domain(
        self,
        domain: str,
        limit: int = 100,
        since: datetime | None = None,
    ) -> list[PreferencePair]:
        """
        List preference pairs by domain for this profile.

        Args:
            domain: Domain to filter by
            limit: Maximum records to return
            since: Only return pairs after this time

        Returns:
            List of preference pairs
        """
        with self._session_factory() as session:
            stmt = select(PreferencePair).where(
                PreferencePair.profile_id == self.profile_id,
                PreferencePair.domain == domain,
            )

            if since:
                stmt = stmt.where(PreferencePair.created_at >= since)

            stmt = stmt.order_by(PreferencePair.created_at.desc()).limit(limit)
            objects = list(session.scalars(stmt).all())
            return cast(list[PreferencePair], detach_all(objects, session))

    def list_domains(self) -> list[str]:
        """
        List all unique domains for this profile.

        Returns:
            List of domain names
        """
        with self._session_factory() as session:
            stmt = (
                select(PreferencePair.domain)
                .where(
                    PreferencePair.profile_id == self.profile_id,
                    PreferencePair.domain.isnot(None),
                )
                .distinct()
                .order_by(PreferencePair.domain)
            )
            return [d for d in session.scalars(stmt).all() if d]

    def count_by_domain(self) -> dict[str, int]:
        """
        Count preference pairs by domain for this profile.

        Returns:
            Dict mapping domain to count
        """
        with self._session_factory() as session:
            domains = self.list_domains()
            counts = {}
            for domain in domains:
                stmt = select(PreferencePair).where(
                    PreferencePair.profile_id == self.profile_id,
                    PreferencePair.domain == domain,
                )
                counts[domain] = len(list(session.scalars(stmt).all()))

            # Count null domain
            stmt = select(PreferencePair).where(
                PreferencePair.profile_id == self.profile_id,
                PreferencePair.domain.is_(None),
            )
            null_count = len(list(session.scalars(stmt).all()))
            if null_count > 0:
                counts[None] = null_count  # type: ignore

            return counts
