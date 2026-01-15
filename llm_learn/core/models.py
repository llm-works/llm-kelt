"""SQLAlchemy ORM models for Learn framework."""

from datetime import date, datetime
from typing import TYPE_CHECKING, Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

if TYPE_CHECKING:
    pass

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

    A profile contains all context injection data for a specific context:
    facts, feedback, preferences, etc. Identified by workspace_slug/profile_slug.
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
    facts: Mapped[list["Fact"]] = relationship(back_populates="profile")
    content: Mapped[list["Content"]] = relationship(back_populates="profile")
    feedback: Mapped[list["Feedback"]] = relationship(back_populates="profile")
    preference_pairs: Mapped[list["PreferencePair"]] = relationship(back_populates="profile")
    interactions: Mapped[list["Interaction"]] = relationship(back_populates="profile")
    predictions: Mapped[list["Prediction"]] = relationship(back_populates="profile")
    directives: Mapped[list["Directive"]] = relationship(back_populates="profile")

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
# Data Models (all associated with a Profile)
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
    # Embedding column - Vector(1536) for OpenAI embeddings, nullable for now
    # Created via migration with pgvector extension
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )
    fetched_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )

    # Relationships
    profile: Mapped["Profile"] = relationship(back_populates="content")
    feedback: Mapped[list["Feedback"]] = relationship(back_populates="content")
    interactions: Mapped[list["Interaction"]] = relationship(back_populates="content")

    __table_args__ = (
        # Hash unique within profile (same content can exist in different profiles)
        UniqueConstraint("profile_id", "content_hash", name="uq_content_profile_hash"),
        Index("idx_content_profile", "profile_id"),
        Index("idx_content_source", "source"),
        Index("idx_content_created", "created_at"),
        Index("idx_content_external_id", "external_id"),
    )

    def __repr__(self) -> str:
        return f"<Content(id={self.id}, source={self.source!r}, hash={self.content_hash[:8]}...)>"


class Feedback(Base):
    """
    Explicit user feedback on content.

    Signals: positive, negative, dismiss
    Strength: 0.0-1.0 (default 1.0)
    """

    __tablename__ = "feedback"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    profile_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("profiles.id"), nullable=False)
    content_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("content.id"), nullable=True
    )
    signal: Mapped[str] = mapped_column(String(20), nullable=False)  # positive, negative, dismiss
    strength: Mapped[float] = mapped_column(Float, default=1.0, nullable=False)
    tags: Mapped[list[str] | None] = mapped_column(ARRAY(String(50)), nullable=True)
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    context: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )

    # Relationships
    profile: Mapped["Profile"] = relationship(back_populates="feedback")
    content: Mapped[Optional["Content"]] = relationship(back_populates="feedback")

    __table_args__ = (
        Index("idx_feedback_profile", "profile_id"),
        Index("idx_feedback_signal", "signal"),
        Index("idx_feedback_created", "created_at"),
        Index("idx_feedback_content_id", "content_id"),
    )

    def __repr__(self) -> str:
        return f"<Feedback(id={self.id}, signal={self.signal!r}, strength={self.strength})>"


class PreferencePair(Base):
    """
    Preference pairs for DPO training.

    Stores chosen vs rejected responses for a given context.
    """

    __tablename__ = "preference_pairs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    profile_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("profiles.id"), nullable=False)
    context: Mapped[str] = mapped_column(Text, nullable=False)
    chosen: Mapped[str] = mapped_column(Text, nullable=False)
    rejected: Mapped[str] = mapped_column(Text, nullable=False)
    margin: Mapped[float | None] = mapped_column(Float, nullable=True)
    domain: Mapped[str | None] = mapped_column(String(100), nullable=True)
    metadata_: Mapped[dict | None] = mapped_column("metadata", JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )

    # Relationships
    profile: Mapped["Profile"] = relationship(back_populates="preference_pairs")

    __table_args__ = (
        Index("idx_preference_pairs_profile", "profile_id"),
        Index("idx_preference_pairs_domain", "domain"),
        Index("idx_preference_pairs_created", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<PreferencePair(id={self.id}, domain={self.domain!r})>"


class Interaction(Base):
    """
    Implicit interaction signals.

    Types: view, click, read, scroll, dismiss
    """

    __tablename__ = "interactions"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    profile_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("profiles.id"), nullable=False)
    content_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("content.id"), nullable=True
    )
    interaction_type: Mapped[str] = mapped_column(String(50), nullable=False)
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    scroll_depth: Mapped[float | None] = mapped_column(Float, nullable=True)
    context: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )

    # Relationships
    profile: Mapped["Profile"] = relationship(back_populates="interactions")
    content: Mapped[Optional["Content"]] = relationship(back_populates="interactions")

    __table_args__ = (
        Index("idx_interactions_profile", "profile_id"),
        Index("idx_interactions_content", "content_id"),
        Index("idx_interactions_type", "interaction_type"),
        Index("idx_interactions_created", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<Interaction(id={self.id}, type={self.interaction_type!r})>"


class Prediction(Base):
    """
    Hypothesis tracking for calibration.

    Records predictions with confidence levels and tracks outcomes.
    """

    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    profile_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("profiles.id"), nullable=False)
    hypothesis: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    confidence_reasoning: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Resolution criteria
    resolution_type: Mapped[str | None] = mapped_column(
        String(50), nullable=True
    )  # date, event, metric
    resolution_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    resolution_event: Mapped[str | None] = mapped_column(Text, nullable=True)
    resolution_metric: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    # Verification
    verification_source: Mapped[str | None] = mapped_column(String(100), nullable=True)
    verification_url: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Outcome
    status: Mapped[str] = mapped_column(
        String(20), default="pending", nullable=False
    )  # pending, resolved
    outcome: Mapped[str | None] = mapped_column(
        String(20), nullable=True
    )  # correct, incorrect, partial, cancelled
    outcome_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    actual_result: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Metadata
    domain: Mapped[str | None] = mapped_column(String(100), nullable=True)
    tags: Mapped[list[str] | None] = mapped_column(ARRAY(String(50)), nullable=True)
    related_content_ids: Mapped[list[int] | None] = mapped_column(ARRAY(BigInteger), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Relationships
    profile: Mapped["Profile"] = relationship(back_populates="predictions")

    __table_args__ = (
        Index("idx_predictions_profile", "profile_id"),
        Index("idx_predictions_status", "status"),
        Index("idx_predictions_domain", "domain"),
        Index("idx_predictions_resolution_date", "resolution_date"),
        Index("idx_predictions_created", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<Prediction(id={self.id}, status={self.status!r}, confidence={self.confidence})>"


class Directive(Base):
    """
    Standing user directives/goals.

    Types: standing, one-time, rule
    Strictly per-profile.
    """

    __tablename__ = "directives"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    profile_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("profiles.id"), nullable=False)
    directive_text: Mapped[str] = mapped_column(Text, nullable=False)
    directive_type: Mapped[str | None] = mapped_column(
        String(50), nullable=True
    )  # standing, one-time, rule
    parsed_rules: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    status: Mapped[str] = mapped_column(
        String(20), default="active", nullable=False
    )  # active, paused, completed
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Relationships
    profile: Mapped["Profile"] = relationship(back_populates="directives")

    __table_args__ = (
        Index("idx_directives_profile", "profile_id"),
        Index("idx_directives_status", "status"),
        Index("idx_directives_created", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<Directive(id={self.id}, type={self.directive_type!r}, status={self.status!r})>"


class Fact(Base):
    """
    User facts for context injection.

    Stores facts about the user that get injected into the system prompt
    at query time. Categories: preferences, background, rules, etc.
    Source: user (explicit), inferred (from conversations), system.
    Strictly per-profile.
    """

    __tablename__ = "facts"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    profile_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("profiles.id"), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    category: Mapped[str | None] = mapped_column(String(50), nullable=True)
    source: Mapped[str] = mapped_column(String(50), default="user", nullable=False)
    confidence: Mapped[float] = mapped_column(Float, default=1.0, nullable=False)
    active: Mapped[bool] = mapped_column(default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )
    updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Relationships
    profile: Mapped["Profile"] = relationship(back_populates="facts")
    embeddings: Mapped[list["FactEmbedding"]] = relationship(back_populates="fact")

    __table_args__ = (
        Index("idx_facts_profile", "profile_id"),
        Index("idx_facts_category", "category"),
        Index("idx_facts_active", "active"),
        Index("idx_facts_created", "created_at"),
    )

    def __repr__(self) -> str:
        preview = self.content[:30] + "..." if len(self.content) > 30 else self.content
        return f"<Fact(id={self.id}, category={self.category!r}, content={preview!r})>"


class FactEmbedding(Base):
    """
    Embeddings for facts, stored separately for model flexibility.

    Allows multiple embedding models per fact and easy model upgrades.
    The current/active model is defined in config, not per-row.
    """

    __tablename__ = "fact_embeddings"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    fact_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("facts.id"), nullable=False)
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    dimensions: Mapped[int] = mapped_column(Integer, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(), nullable=False)  # Unconstrained
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )

    # Relationships
    fact: Mapped["Fact"] = relationship(back_populates="embeddings")

    __table_args__ = (
        # Only one embedding per fact per model
        UniqueConstraint("fact_id", "model_name", name="uq_fact_embedding_model"),
        Index("idx_fact_embeddings_fact", "fact_id"),
        Index("idx_fact_embeddings_model", "model_name"),
    )
