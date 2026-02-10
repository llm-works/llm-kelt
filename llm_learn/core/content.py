"""Content model and store - raw ingested content."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from datetime import datetime
from typing import Any, cast

from appinfra.db.utils import detach, detach_all
from sqlalchemy import DateTime, Index, String, Text, UniqueConstraint, select, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base, utc_now
from .exceptions import ValidationError


class Content(Base):
    """
    Raw ingested content for reference.

    Content is the central reference point - feedback, interactions,
    and other signals reference content by ID. Isolated by context_key.

    Content is deduplicated by hash within a context - if the same text
    is ingested twice, only one record is created.
    """

    __tablename__ = "content"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    context_key: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
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

    __table_args__ = (
        UniqueConstraint("context_key", "content_hash", name="uq_content_context_hash"),
        Index("idx_content_context", "context_key"),
        Index("idx_content_source", "source"),
        Index("idx_content_created", "created_at"),
        Index("idx_content_external_id", "external_id"),
        # Prefix index for efficient LIKE queries (pattern matching)
        Index(
            "idx_content_context_prefix",
            "context_key",
            postgresql_ops={"context_key": "text_pattern_ops"},
        ),
        # Partial unique index for NULL context_key to ensure deduplication works
        # When context_key IS NULL, content_hash must be unique (global scope deduplication)
        Index(
            "uq_content_null_context_hash",
            "content_hash",
            unique=True,
            postgresql_where=text("context_key IS NULL"),
        ),
    )

    def __repr__(self) -> str:
        return f"<Content(id={self.id}, source={self.source!r}, hash={self.content_hash[:8]}...)>"


class ContentStore:
    """
    Content storage operations scoped to a context.

    Provides methods for creating, retrieving, and listing content.
    Automatically deduplicates by content hash within the context.
    """

    def __init__(self, session_factory: Callable[[], Any], context_key: str | None) -> None:
        """
        Initialize ContentStore scoped to a context.

        Args:
            session_factory: Callable that returns a context manager for database sessions.
            context_key: Context key to scope all operations to (None = no filtering).
                Supports SQL LIKE patterns (% and _) for prefix/pattern matching.
                Examples:
                  - "acme:prod:reviewer" - exact match
                  - "acme:prod:%" - all profiles in workspace
                  - "acme:%" - all workspaces in domain
        """
        self._session_factory = session_factory
        self._context_key = context_key

    @property
    def context_key(self) -> str | None:
        """Get the context key this store is scoped to."""
        return self._context_key

    def _build_context_filter(self, column):
        """
        Build context filter condition with pattern matching support.

        Args:
            column: SQLAlchemy column to filter on.

        Returns:
            SQLAlchemy filter condition, or None if no filtering needed.
        """
        from llm_learn.memory.isolation import build_context_filter

        return build_context_filter(self._context_key, column)

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
                context_key=self._context_key,
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
            stmt = self._build_hash_query(content_hash)
            existing = session.scalar(stmt)
            if existing:
                return existing.id, False

            return self._create_with_retry(
                session, stmt, content_hash, content_text, source, external_id, url, title, metadata
            )

    def get(self, content_id: int) -> Content | None:
        """
        Get content by ID.

        Args:
            content_id: The content ID.

        Returns:
            Content record if found and belongs to profile, None otherwise.
        """
        with self._session_factory() as session:
            stmt = select(Content).where(Content.id == content_id)

            # Apply context filter (supports pattern matching)
            context_filter = self._build_context_filter(Content.context_key)
            if context_filter is not None:
                stmt = stmt.where(context_filter)

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
            stmt = select(Content).where(Content.content_hash == content_hash)

            # Apply context filter (supports pattern matching)
            context_filter = self._build_context_filter(Content.context_key)
            if context_filter is not None:
                stmt = stmt.where(context_filter)

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
            stmt = select(Content).where(Content.external_id == external_id)

            # Apply context filter (supports pattern matching)
            context_filter = self._build_context_filter(Content.context_key)
            if context_filter is not None:
                stmt = stmt.where(context_filter)

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
            stmt = select(Content)

            # Apply context filter (supports pattern matching)
            context_filter = self._build_context_filter(Content.context_key)
            if context_filter is not None:
                stmt = stmt.where(context_filter)

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
            stmt = select(func.count()).select_from(Content)

            # Apply context filter (supports pattern matching)
            context_filter = self._build_context_filter(Content.context_key)
            if context_filter is not None:
                stmt = stmt.where(context_filter)

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
            stmt = select(Content).where(Content.id == content_id)

            # Apply context filter (supports pattern matching)
            context_filter = self._build_context_filter(Content.context_key)
            if context_filter is not None:
                stmt = stmt.where(context_filter)

            content = session.scalar(stmt)
            if content:
                session.delete(content)
                return True
            return False

    def _build_hash_query(self, content_hash: str):
        """Build query to find content by hash with context filter."""
        stmt = select(Content).where(Content.content_hash == content_hash)
        context_filter = self._build_context_filter(Content.context_key)
        if context_filter is not None:
            stmt = stmt.where(context_filter)
        return stmt

    def _create_with_retry(
        self, session, stmt, content_hash, content_text, source, external_id, url, title, metadata
    ) -> tuple[int, bool]:
        """Create content with retry on integrity error."""
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
            context_key=self._context_key,
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
