"""Workspace model - logical grouping within a domain."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, ForeignKey, Index, String, Text, UniqueConstraint, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base, generate_id, utc_now

if TYPE_CHECKING:
    from .domain import Domain
    from .profile import Profile


class Workspace(Base):
    """
    Logical grouping within a domain (project/team/environment).

    A workspace groups profiles that share common configuration or purpose.
    Workspaces can optionally belong to a domain, or exist independently
    for simpler single-tenant setups.

    The ID is a hash computed from the domain slug and workspace slug at creation time.
    Once created, the ID remains stable even if slugs change.

    Examples:
        - "default" workspace for personal use
        - "development", "staging", "production" for different environments
        - "team-alpha", "team-beta" for team-based isolation
    """

    __tablename__ = "workspaces"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    domain_id: Mapped[str | None] = mapped_column(
        String(32), ForeignKey("domains.id"), nullable=True
    )
    slug: Mapped[str] = mapped_column(String(50), nullable=False)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    config: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )

    # Relationships
    domain: Mapped[Domain | None] = relationship(back_populates="workspaces")
    profiles: Mapped[list[Profile]] = relationship(back_populates="workspace")

    __table_args__ = (
        # Slug unique within domain (NULL domain treated separately by DB)
        UniqueConstraint("domain_id", "slug", name="uq_workspace_domain_slug"),
        # Partial unique index for domain-less workspaces (prevents duplicate slugs when domain is NULL)
        Index(
            "uq_workspace_slug_null_domain",
            "slug",
            unique=True,
            postgresql_where=text("domain_id IS NULL"),
        ),
        Index("idx_workspaces_domain", "domain_id"),
        Index("idx_workspaces_slug", "slug"),
    )

    @staticmethod
    def generate_id(domain_slug: str | None, workspace_slug: str) -> str:
        """Generate hash-based ID from domain and workspace slugs."""
        return generate_id("workspace", domain_slug, workspace_slug)

    @property
    def identifier(self) -> str:
        """Full path: domain/workspace or just workspace if no domain."""
        if self.domain:
            return f"{self.domain.slug}/{self.slug}"
        return self.slug

    def __repr__(self) -> str:
        return f"<Workspace(id={self.id[:8]}..., identifier={self.identifier!r})>"
