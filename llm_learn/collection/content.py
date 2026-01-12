"""Content storage client."""

import hashlib
from typing import cast

from appinfra.db.utils import detach, detach_all
from sqlalchemy import select

from ..core.exceptions import ValidationError
from ..core.models import Content
from .base import ProfileScopedClient


class ContentClient(ProfileScopedClient[Content]):
    """
    Client for managing content storage scoped to a profile.

    Content is the central reference for feedback and interactions.
    Automatically deduplicates by content hash within a profile.
    """

    model = Content

    def create(
        self,
        content_text: str,
        source: str,
        external_id: str | None = None,
        url: str | None = None,
        title: str | None = None,
        metadata: dict | None = None,
    ) -> int:
        """
        Create a new content record.

        Args:
            content_text: Full text content
            source: Source identifier (e.g., "hn", "arxiv", "feedback")
            external_id: Original ID from source
            url: URL of content
            title: Title of content
            metadata: Additional metadata

        Returns:
            Created content ID

        Raises:
            ValidationError: If content_text is empty
        """
        if not content_text or not content_text.strip():
            raise ValidationError("Content text cannot be empty")

        if not source or not source.strip():
            raise ValidationError("Source cannot be empty")

        content_hash = self._compute_hash(content_text)

        with self._session_factory() as session:
            content = Content(
                profile_id=self.profile_id,
                external_id=external_id,
                source=source.strip(),
                url=url,
                title=title,
                content_text=content_text,
                content_hash=content_hash,
                metadata_=metadata,
            )
            session.add(content)
            session.flush()  # Get ID before commit
            return content.id

    def get_or_create(
        self,
        content_text: str,
        source: str,
        external_id: str | None = None,
        url: str | None = None,
        title: str | None = None,
        metadata: dict | None = None,
    ) -> tuple[int, bool]:
        """
        Get existing content by hash or create new (within this profile).

        Args:
            content_text: Full text content
            source: Source identifier
            external_id: Original ID from source
            url: URL of content
            title: Title of content
            metadata: Additional metadata

        Returns:
            Tuple of (content_id, created) where created is True if new
        """
        if not content_text or not content_text.strip():
            raise ValidationError("Content text cannot be empty")

        content_hash = self._compute_hash(content_text)

        with self._session_factory() as session:
            # Try to find existing by hash within this profile
            stmt = select(Content).where(
                Content.profile_id == self.profile_id,
                Content.content_hash == content_hash,
            )
            existing = session.scalar(stmt)
            if existing:
                return existing.id, False

            # Create new
            content = Content(
                profile_id=self.profile_id,
                external_id=external_id,
                source=source.strip() if source else "unknown",
                url=url,
                title=title,
                content_text=content_text,
                content_hash=content_hash,
                metadata_=metadata,
            )
            session.add(content)
            session.flush()
            return content.id, True

    def find_by_hash(self, content_hash: str) -> Content | None:
        """
        Find content by hash within this profile.

        Args:
            content_hash: SHA-256 hash of content text

        Returns:
            Content record if found, None otherwise
        """
        with self._session_factory() as session:
            stmt = select(Content).where(
                Content.profile_id == self.profile_id,
                Content.content_hash == content_hash,
            )
            obj = session.scalar(stmt)
            if obj:
                return cast(Content, detach(obj, session))
            return None

    def find_by_external_id(self, external_id: str, source: str | None = None) -> Content | None:
        """
        Find content by external ID within this profile.

        Args:
            external_id: Original ID from source
            source: Optional source filter

        Returns:
            Content record if found, None otherwise
        """
        with self._session_factory() as session:
            stmt = select(Content).where(
                Content.profile_id == self.profile_id,
                Content.external_id == external_id,
            )
            if source:
                stmt = stmt.where(Content.source == source)
            obj = session.scalar(stmt)
            if obj:
                return cast(Content, detach(obj, session))
            return None

    def list_by_source(self, source: str, limit: int = 100) -> list[Content]:
        """
        List content from a specific source within this profile.

        Args:
            source: Source identifier
            limit: Maximum records to return

        Returns:
            List of content records
        """
        with self._session_factory() as session:
            stmt = (
                select(Content)
                .where(
                    Content.profile_id == self.profile_id,
                    Content.source == source,
                )
                .order_by(Content.created_at.desc())
                .limit(limit)
            )
            objects = list(session.scalars(stmt).all())
            return cast(list[Content], detach_all(objects, session))

    @staticmethod
    def _compute_hash(content_text: str) -> str:
        """Compute SHA-256 hash of content text."""
        return hashlib.sha256(content_text.encode("utf-8")).hexdigest()
