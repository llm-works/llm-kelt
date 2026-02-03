"""Profile model - individual context."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import Boolean, DateTime, ForeignKey, Index, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base, generate_id, utc_now

if TYPE_CHECKING:
    from .content import Content
    from .workspace import Workspace


class Profile(Base):
    """
    Individual context (user/agent/persona).

    A profile contains all context injection data for a specific identity.
    Memory data (facts, predictions, etc.) is scoped to a profile.

    The ID is a hash computed from the full path (domain/workspace/profile)
    at creation time. Once created, the ID remains stable even if slugs change.

    Examples:
        - Personal profile: "default/me"
        - Agent profile: "production/code-reviewer"
        - User profile: "acme/research/alice"
    """

    __tablename__ = "profiles"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    workspace_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("workspaces.id"), nullable=False
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
    workspace: Mapped[Workspace] = relationship(back_populates="profiles")
    content: Mapped[list[Content]] = relationship(back_populates="profile")

    __table_args__ = (
        UniqueConstraint("workspace_id", "slug", name="uq_profile_workspace_slug"),
        Index("idx_profiles_workspace", "workspace_id"),
        Index("idx_profiles_slug", "slug"),
        Index("idx_profiles_active", "active"),
    )

    @staticmethod
    def generate_id(domain_slug: str | None, workspace_slug: str, profile_slug: str) -> str:
        """Generate hash-based ID from full path."""
        return generate_id("profile", domain_slug, workspace_slug, profile_slug)

    @property
    def identifier(self) -> str:
        """Full path: domain/workspace/profile or workspace/profile."""
        ws = self.workspace
        if ws.domain:
            return f"{ws.domain.slug}/{ws.slug}/{self.slug}"
        return f"{ws.slug}/{self.slug}"

    def __repr__(self) -> str:
        return f"<Profile(id={self.id[:8]}..., identifier={self.identifier!r})>"
