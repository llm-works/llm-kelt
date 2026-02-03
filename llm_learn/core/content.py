"""Content model and store - raw ingested content."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from datetime import datetime
from typing import TYPE_CHECKING, Any, cast

from appinfra.db.utils import detach, detach_all
from sqlalchemy import DateTime, ForeignKey, Index, String, Text, UniqueConstraint, select
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base, utc_now
from .exceptions import ValidationError

if TYPE_CHECKING:
    from .profile import Profile


class Content(Base):
    """
    Raw ingested content for reference.

    Content is the central reference point - feedback, interactions,
    and other signals reference content by ID. Strictly per-profile.

    Content is deduplicated by hash within a profile - if the same text
    is ingested twice, only one record is created.
    """

    __tablename__ = "content"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    profile_id: Mapped[str] = mapped_column(String(32), ForeignKey("profiles.id"), nullable=False)
    external_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    source: Mapped[str] = mapped_column(String(100), nullable=False)
    url: Mapped[str | None] = mapped_column(Text, nullable=True)
    title: Mapped[str | None] = mapped_column(Text, nullable=True)
    content_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    metadata_: Mapped[dict | None] = mapped_column("metadata", JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )
    fetched_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )

    # Relationships
    profile: Mapped[Profile] = relationship(back_populates="content")

    __table_args__ = (
        UniqueConstraint("profile_id", "content_hash", name="uq_content_profile_hash"),
        Index("idx_content_profile", "profile_id"),
        Index("idx_content_source", "source"),
        Index("idx_content_created", "created_at"),
        Index("idx_content_external_id", "external_id"),
    )

    def __repr__(self) -> str:
        return f"<Content(id={self.id}, source={self.source!r}, hash={self.content_hash[:8]}...)>"


class ContentStore:
    """
    Content storage operations scoped to a profile.

    Provides methods for creating, retrieving, and listing content.
    Automatically deduplicates by content hash within the profile.
    """

    def __init__(self, session_factory: Callable[[], Any], profile_id: str) -> None:
        """
        Initialize ContentStore scoped to a profile.

        Args:
            session_factory: Callable that returns a context manager for database sessions.
            profile_id: Profile ID (32-char hash) to scope all operations to.
        """
        self._session_factory = session_factory
        self._profile_id = profile_id

    @property
    def profile_id(self) -> str:
        """Get the profile ID this store is scoped to."""
        return self._profile_id

    def create(
        self,
        content_text: str,
        source: str,
        *,
        external_id: str | None = None,
        url: str | None = None,
        title: str | None = None,
        metadata: dict | None = None,
    ) -> int:
        """
        Create a new content record.

        Args:
            content_text: Full text content.
            source: Source identifier (e.g., "hn", "arxiv", "feedback").
            external_id: Original ID from source.
            url: URL of content.
            title: Title of content.
            metadata: Additional metadata.

        Returns:
            Created content ID.

        Raises:
            ValidationError: If content_text or source is empty.
        """
        if not content_text or not content_text.strip():
            raise ValidationError("Content text cannot be empty")
        if not source or not source.strip():
            raise ValidationError("Source cannot be empty")

        content_hash = self._compute_hash(content_text)

        with self._session_factory() as session:
            content = Content(
                profile_id=self._profile_id,
                external_id=external_id,
                source=source.strip(),
                url=url,
                title=title,
                content_text=content_text,
                content_hash=content_hash,
                metadata_=metadata,
            )
            session.add(content)
            session.flush()
            return content.id

    def get_or_create(
        self,
        content_text: str,
        source: str,
        *,
        external_id: str | None = None,
        url: str | None = None,
        title: str | None = None,
        metadata: dict | None = None,
    ) -> tuple[int, bool]:
        """
        Get existing content by hash or create new.

        Args:
            content_text: Full text content.
            source: Source identifier.
            external_id: Original ID from source.
            url: URL of content.
            title: Title of content.
            metadata: Additional metadata.

        Returns:
            Tuple of (content_id, created) where created is True if new.

        Raises:
            ValidationError: If content_text or source is empty.
        """
        if not content_text or not content_text.strip():
            raise ValidationError("Content text cannot be empty")
        if not source or not source.strip():
            raise ValidationError("Source cannot be empty")

        content_hash = self._compute_hash(content_text)

        with self._session_factory() as session:
            stmt = select(Content).where(
                Content.profile_id == self._profile_id,
                Content.content_hash == content_hash,
            )
            existing = session.scalar(stmt)
            if existing:
                return existing.id, False

            content = self._create_content_record(
                content_hash, content_text, source, external_id, url, title, metadata
            )
            try:
                session.add(content)
                session.flush()
                return content.id, True
            except IntegrityError:
                session.rollback()
                existing = session.scalar(stmt)
                if existing:
                    return existing.id, False
                raise

    def get(self, content_id: int) -> Content | None:
        """
        Get content by ID.

        Args:
            content_id: The content ID.

        Returns:
            Content record if found and belongs to profile, None otherwise.
        """
        with self._session_factory() as session:
            stmt = select(Content).where(
                Content.id == content_id,
                Content.profile_id == self._profile_id,
            )
            obj = session.scalar(stmt)
            if obj:
                return cast(Content, detach(obj, session))
            return None

    def find_by_hash(self, content_hash: str) -> Content | None:
        """
        Find content by hash within this profile.

        Args:
            content_hash: SHA-256 hash of content text.

        Returns:
            Content record if found, None otherwise.
        """
        with self._session_factory() as session:
            stmt = select(Content).where(
                Content.profile_id == self._profile_id,
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
            external_id: Original ID from source.
            source: Optional source filter.

        Returns:
            Content record if found, None otherwise.
        """
        with self._session_factory() as session:
            stmt = select(Content).where(
                Content.profile_id == self._profile_id,
                Content.external_id == external_id,
            )
            if source:
                stmt = stmt.where(Content.source == source)
            obj = session.scalar(stmt)
            if obj:
                return cast(Content, detach(obj, session))
            return None

    def list(
        self,
        *,
        source: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Content]:
        """
        List content for this profile.

        Args:
            source: Optional source filter.
            limit: Maximum records to return.
            offset: Records to skip.

        Returns:
            List of content records.
        """
        with self._session_factory() as session:
            stmt = select(Content).where(Content.profile_id == self._profile_id)
            if source:
                stmt = stmt.where(Content.source == source)
            stmt = stmt.order_by(Content.created_at.desc()).limit(limit).offset(offset)
            objects = list(session.scalars(stmt).all())
            return cast(list[Content], detach_all(objects, session))

    def count(self, *, source: str | None = None) -> int:
        """
        Count content for this profile.

        Args:
            source: Optional source filter.

        Returns:
            Total count.
        """
        from sqlalchemy import func

        with self._session_factory() as session:
            stmt = (
                select(func.count())
                .select_from(Content)
                .where(Content.profile_id == self._profile_id)
            )
            if source:
                stmt = stmt.where(Content.source == source)
            return session.scalar(stmt) or 0

    def delete(self, content_id: int) -> bool:
        """
        Delete content by ID.

        Args:
            content_id: The content ID to delete.

        Returns:
            True if deleted, False if not found.
        """
        with self._session_factory() as session:
            stmt = select(Content).where(
                Content.id == content_id,
                Content.profile_id == self._profile_id,
            )
            content = session.scalar(stmt)
            if content:
                session.delete(content)
                return True
            return False

    def _create_content_record(
        self,
        content_hash: str,
        content_text: str,
        source: str,
        external_id: str | None,
        url: str | None,
        title: str | None,
        metadata: dict | None,
    ) -> Content:
        """Build a Content record with the given parameters."""
        return Content(
            profile_id=self._profile_id,
            external_id=external_id,
            source=source.strip(),
            url=url,
            title=title,
            content_text=content_text,
            content_hash=content_hash,
            metadata_=metadata,
        )

    @staticmethod
    def _compute_hash(content_text: str) -> str:
        """Compute SHA-256 hash of content text."""
        return hashlib.sha256(content_text.encode("utf-8")).hexdigest()
