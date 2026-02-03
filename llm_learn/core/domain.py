"""Domain model - top-level isolation boundary."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, Index, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base, generate_id, utc_now

if TYPE_CHECKING:
    from .workspace import Workspace


class Domain(Base):
    """
    Top-level isolation boundary (tenant/org/deployment).

    A domain is the highest level of data isolation. Workspaces belong to domains,
    and all data is transitively scoped through the domain -> workspace -> profile chain.

    Domains are optional - workspaces can exist without a domain for simpler setups.

    The ID is a hash computed from the slug at creation time. Once created,
    the ID remains stable even if the slug is changed.
    """

    __tablename__ = "domains"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    slug: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    config: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )

    # Relationships
    workspaces: Mapped[list[Workspace]] = relationship(back_populates="domain")

    __table_args__ = (Index("idx_domains_slug", "slug"),)

    @staticmethod
    def generate_id(slug: str) -> str:
        """Generate hash-based ID from domain slug."""
        return generate_id("domain", slug)

    def __repr__(self) -> str:
        return f"<Domain(id={self.id[:8]}..., slug={self.slug!r})>"
