"""SQLAlchemy ORM models for Learn framework.

Core organizational models: Workspace, Profile, Content.
Memory models (facts, predictions, etc.) are in memory/v1/models.py.
"""

from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from .utils import utc_now


class Base(DeclarativeBase):
    """Base class for all Learn models."""

    pass


# =============================================================================
# Workspace & Profile (top-level organization)
# =============================================================================


class Workspace(Base):
    """
    Container for related profiles.

    A workspace groups profiles that share common configuration or purpose.
    Examples: "default", "development", "production"
    """

    __tablename__ = "workspaces"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    slug: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    config: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )

    # Relationships
    profiles: Mapped[list["Profile"]] = relationship(back_populates="workspace")

    __table_args__ = (Index("idx_workspaces_slug", "slug"),)

    def __repr__(self) -> str:
        return f"<Workspace(id={self.id}, slug={self.slug!r})>"


class Profile(Base):
    """
    Individual context injection profile.

    A profile contains all context injection data for a specific context.
    Memory data (facts, predictions, etc.) is in memv1_* tables.
    """

    __tablename__ = "profiles"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    workspace_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("workspaces.id"), nullable=False
    )
    slug: Mapped[str] = mapped_column(String(50), nullable=False)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    config: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )

    # Relationships
    workspace: Mapped["Workspace"] = relationship(back_populates="profiles")
    content: Mapped[list["Content"]] = relationship(back_populates="profile")

    __table_args__ = (
        UniqueConstraint("workspace_id", "slug", name="uq_profile_workspace_slug"),
        Index("idx_profiles_workspace", "workspace_id"),
        Index("idx_profiles_slug", "slug"),
        Index("idx_profiles_active", "active"),
    )

    @property
    def identifier(self) -> str:
        """Full identifier: workspace_slug/profile_slug."""
        if self.workspace:
            return f"{self.workspace.slug}/{self.slug}"
        return self.slug

    def __repr__(self) -> str:
        return f"<Profile(id={self.id}, identifier={self.identifier!r})>"


# =============================================================================
# Content (external content storage, not part of memory v1)
# =============================================================================


class Content(Base):
    """
    Stores ingested content for reference and training.

    Content is the central reference point - feedback, interactions,
    and other signals reference content by ID. Strictly per-profile.
    """

    __tablename__ = "content"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    profile_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("profiles.id"), nullable=False)
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
    profile: Mapped["Profile"] = relationship(back_populates="content")

    __table_args__ = (
        UniqueConstraint("profile_id", "content_hash", name="uq_content_profile_hash"),
        Index("idx_content_profile", "profile_id"),
        Index("idx_content_source", "source"),
        Index("idx_content_created", "created_at"),
        Index("idx_content_external_id", "external_id"),
    )

    def __repr__(self) -> str:
        return f"<Content(id={self.id}, source={self.source!r}, hash={self.content_hash[:8]}...)>"
