# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-kelt Authors

"""Knowledge Graph models - entity-centric knowledge with scoped subgraphs.

All KG objects live in a single graph. Scope keys define subgraphs:
- "global" — visible to all
- "org:acme" — visible to org and descendants
- "org:acme:user:alice" — visible to alice (and inherits from ancestors)

When querying with scope S, you see S + all ancestor scopes up to global.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import (
    BigInteger,
    DateTime,
    Float,
    ForeignKey,
    Index,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from llm_kelt.core.base import Base

if TYPE_CHECKING:
    pass

__all__ = [
    "Entity",
    "EntityAlias",
    "EntityRef",
    "EntityRelationship",
    "FactEntity",
    "ScopedMixin",
]


class ScopedMixin:
    """Mixin for scoped KG tables with extra.

    Provides:
    - scope_key: Hierarchical scope for visibility
    - extra: JSONB for extensible attributes

    Each model should add scope indexes in __table_args__:
        Index("ix_{tablename}_scope", "scope_key"),
        Index("ix_{tablename}_scope_prefix", "scope_key",
              postgresql_ops={"scope_key": "varchar_pattern_ops"}),
    """

    scope_key: Mapped[str] = mapped_column(String(255), nullable=False)
    extra: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)


class Entity(ScopedMixin, Base):
    """Canonical entity - a persistent thing that facts can reference.

    Entities have identity-based deduplication via canonical_name + entity_type
    within a scope. Use EntityAlias for alternative names.

    Scope determines visibility:
    - "global" — public, visible to all scopes
    - "org:acme" — visible to org:acme and all descendant scopes
    - "org:acme:user:alice" — private to alice's scope

    Queries resolve hierarchically: scope + ancestors up to global.
    """

    __tablename__ = "kg_entities"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)

    canonical_name: Mapped[str] = mapped_column(String(255), nullable=False)
    entity_type: Mapped[str] = mapped_column(String(50), nullable=False)

    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )
    updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    aliases: Mapped[list[EntityAlias]] = relationship(
        back_populates="entity", cascade="all, delete-orphan"
    )
    fact_links: Mapped[list[FactEntity]] = relationship(back_populates="entity")
    refs: Mapped[list[EntityRef]] = relationship(back_populates="entity")

    __table_args__ = (
        UniqueConstraint(
            "scope_key", "canonical_name", "entity_type", name="uq_kg_entity_identity"
        ),
        Index("ix_kg_entities_scope", "scope_key"),
        Index("ix_kg_entities_scope_type", "scope_key", "entity_type"),
        Index(
            "ix_kg_entities_scope_prefix",
            "scope_key",
            postgresql_ops={"scope_key": "varchar_pattern_ops"},
        ),
    )

    def __repr__(self) -> str:
        return f"<Entity(id={self.id}, name={self.canonical_name!r}, type={self.entity_type!r})>"


class EntityAlias(ScopedMixin, Base):
    """Alternative name for an entity - used for resolution and dedup.

    Aliases enable identity-based deduplication: "Tesla", "TSLA", "Tesla Inc"
    all resolve to the same canonical entity.

    Aliases are scoped. Resolution checks the query scope first, then walks
    up the hierarchy to global.
    """

    __tablename__ = "kg_entity_aliases"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    entity_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("kg_entities.id", ondelete="CASCADE"), nullable=False
    )

    alias: Mapped[str] = mapped_column(String(255), nullable=False)
    alias_normalized: Mapped[str] = mapped_column(String(255), nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )

    entity: Mapped[Entity] = relationship(back_populates="aliases")

    __table_args__ = (
        UniqueConstraint("scope_key", "alias_normalized", name="uq_kg_alias_identity"),
        Index("ix_kg_entity_aliases_scope", "scope_key"),
        Index("ix_kg_entity_aliases_entity", "entity_id"),
        Index(
            "ix_kg_entity_aliases_scope_prefix",
            "scope_key",
            postgresql_ops={"scope_key": "varchar_pattern_ops"},
        ),
    )

    def __repr__(self) -> str:
        return f"<EntityAlias(id={self.id}, alias={self.alias!r}, entity_id={self.entity_id})>"


class EntityRelationship(ScopedMixin, Base):
    """Relationship between two entities.

    Relationships are scoped. "Tesla employs Elon Musk" might be global,
    while "Tesla in_my_watchlist" is user-scoped.
    """

    __tablename__ = "kg_entity_relationships"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)

    from_entity_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("kg_entities.id", ondelete="CASCADE"), nullable=False
    )
    to_entity_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("kg_entities.id", ondelete="CASCADE"), nullable=False
    )
    relationship_type: Mapped[str] = mapped_column(String(50), nullable=False)

    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )

    from_entity: Mapped[Entity] = relationship(foreign_keys=[from_entity_id])
    to_entity: Mapped[Entity] = relationship(foreign_keys=[to_entity_id])

    __table_args__ = (
        UniqueConstraint(
            "scope_key",
            "from_entity_id",
            "to_entity_id",
            "relationship_type",
            name="uq_kg_entity_rel",
        ),
        Index("ix_kg_entity_relationships_scope", "scope_key"),
        Index("ix_kg_entity_relationships_from", "from_entity_id"),
        Index("ix_kg_entity_relationships_to", "to_entity_id"),
        Index("ix_kg_entity_relationships_type", "from_entity_id", "relationship_type"),
        Index(
            "ix_kg_entity_relationships_scope_prefix",
            "scope_key",
            postgresql_ops={"scope_key": "varchar_pattern_ops"},
        ),
    )

    def __repr__(self) -> str:
        return (
            f"<EntityRelationship(id={self.id}, "
            f"{self.from_entity_id}-[{self.relationship_type}]->{self.to_entity_id})>"
        )


class FactEntity(ScopedMixin, Base):
    """Links kelt atomic facts to KG entities.

    A fact can reference multiple entities (e.g., "Tesla acquired SolarCity"
    references both Tesla and SolarCity). The role field indicates the
    relationship type (subject, object, mentioned, etc.).
    """

    __tablename__ = "kg_fact_entities"

    fact_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("atomic_facts.id", ondelete="CASCADE"), primary_key=True
    )
    entity_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("kg_entities.id", ondelete="CASCADE"), primary_key=True
    )

    role: Mapped[str] = mapped_column(String(50), nullable=False, default="subject")
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )

    entity: Mapped[Entity] = relationship(back_populates="fact_links")

    __table_args__ = (
        Index("ix_kg_fact_entities_scope", "scope_key"),
        Index("ix_kg_fact_entities_entity", "entity_id"),
        Index("ix_kg_fact_entities_fact", "fact_id"),
        Index(
            "ix_kg_fact_entities_scope_prefix",
            "scope_key",
            postgresql_ops={"scope_key": "varchar_pattern_ops"},
        ),
    )

    def __repr__(self) -> str:
        return (
            f"<FactEntity(fact_id={self.fact_id}, entity_id={self.entity_id}, role={self.role!r})>"
        )


class EntityRef(ScopedMixin, Base):
    """Reference to an entity - lightweight signal for provenance and analytics.

    Tracks where/when entities were referenced. Separate from scope membership:
    - Scope determines visibility (access control)
    - Refs track provenance (where did this come from, with what sentiment, etc.)

    Use cases:
    - Track which sources mentioned an entity
    - Aggregate sentiment signals
    - Build entity importance scores
    """

    __tablename__ = "kg_entity_refs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    entity_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("kg_entities.id", ondelete="CASCADE"), nullable=False
    )

    source_type: Mapped[str] = mapped_column(String(50), nullable=False)
    source_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    source_url: Mapped[str | None] = mapped_column(String(2048), nullable=True)

    snippet: Mapped[str | None] = mapped_column(Text, nullable=True)
    sentiment: Mapped[float | None] = mapped_column(Float, nullable=True)

    ref_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )

    entity: Mapped[Entity] = relationship(back_populates="refs")

    __table_args__ = (
        Index("ix_kg_entity_refs_scope", "scope_key"),
        Index("ix_kg_entity_refs_entity", "entity_id"),
        Index("ix_kg_entity_refs_entity_time", "entity_id", "ref_at"),
        Index(
            "ix_kg_entity_refs_scope_prefix",
            "scope_key",
            postgresql_ops={"scope_key": "varchar_pattern_ops"},
        ),
    )

    def __repr__(self) -> str:
        return f"<EntityRef(id={self.id}, entity_id={self.entity_id}, source={self.source_type!r})>"
