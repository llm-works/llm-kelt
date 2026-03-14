"""Atomic memory model - fact-based knowledge storage.

All knowledge is stored as facts with type-specific detail tables.
Table names are prefixed with atomic_ for schema versioning.

This is a unified model where every piece of knowledge is a "fact" with
a type discriminator and optional type-specific details.
"""

from datetime import UTC, date, datetime

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
    text,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from llm_kelt.core.base import Base

# =============================================================================
# Base Fact Table
# =============================================================================


class Fact(Base):
    """
    Base table for all atomic facts.

    Every piece of knowledge is a fact with a type. Type-specific details
    are stored in separate detail tables linked by fact_id.

    Types:
        - assertion: Simple facts about the user (no detail table needed)
        - solution: Agent problem/answer records
        - prediction: Hypotheses with resolution tracking
        - feedback: Explicit user signals on content
        - directive: Standing instructions/rules
        - interaction: Implicit behavioral signals
        - preference: DPO training pairs
    """

    __tablename__ = "atomic_facts"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    context_key: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    type: Mapped[str] = mapped_column(String(20), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    category: Mapped[str | None] = mapped_column(String(50), nullable=True)
    source: Mapped[str] = mapped_column(String(50), default="user", nullable=False)
    confidence: Mapped[float] = mapped_column(Float, default=1.0, nullable=False)
    active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
    )
    updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Detail table relationships (one-to-one)
    solution_details: Mapped["SolutionDetails | None"] = relationship(
        back_populates="fact", uselist=False, passive_deletes=True
    )
    prediction_details: Mapped["PredictionDetails | None"] = relationship(
        back_populates="fact", uselist=False, passive_deletes=True
    )
    feedback_details: Mapped["FeedbackDetails | None"] = relationship(
        back_populates="fact", uselist=False, passive_deletes=True
    )
    directive_details: Mapped["DirectiveDetails | None"] = relationship(
        back_populates="fact", uselist=False, passive_deletes=True
    )
    interaction_details: Mapped["InteractionDetails | None"] = relationship(
        back_populates="fact", uselist=False, passive_deletes=True
    )
    preference_details: Mapped["PreferenceDetails | None"] = relationship(
        back_populates="fact", uselist=False, passive_deletes=True
    )

    __table_args__ = (
        Index("idx_atomic_facts_context", "context_key"),
        Index("idx_atomic_facts_context_type", "context_key", "type"),
        Index("idx_atomic_facts_category", "category"),
        Index("idx_atomic_facts_context_active", "context_key", "active"),
        Index("idx_atomic_facts_created", "created_at"),
        # Prefix index for efficient LIKE queries (pattern matching)
        Index(
            "idx_atomic_facts_context_prefix",
            "context_key",
            postgresql_ops={"context_key": "varchar_pattern_ops"},
        ),
        # Partial unique index for NULL context_key to ensure deduplication works
        # When context_key IS NULL, (type, content_hash) must be unique per fact type
        Index(
            "uq_atomic_facts_null_context_hash",
            "type",
            "content_hash",
            unique=True,
            postgresql_where=text("context_key IS NULL"),
        ),
    )

    def __repr__(self) -> str:
        preview = self.content[:30] + "..." if len(self.content) > 30 else self.content
        return f"<Fact(id={self.id}, type={self.type!r}, content={preview!r})>"


# =============================================================================
# Detail Tables
# =============================================================================


class SolutionDetails(Base):
    """
    Details for solution facts (agent problem/answer records).

    A solution represents an agent completing a task:
    - problem: What needed to be solved
    - answer: What the agent produced
    """

    __tablename__ = "atomic_solution_details"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    fact_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("atomic_facts.id", ondelete="CASCADE"), nullable=False
    )

    # Problem (input)
    agent_name: Mapped[str] = mapped_column(String(100), nullable=False)
    problem: Mapped[str] = mapped_column(Text, nullable=False)
    problem_context: Mapped[dict] = mapped_column(JSONB, nullable=False)

    # Answer (output)
    answer: Mapped[dict] = mapped_column(JSONB, nullable=False)
    answer_text: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Performance metrics
    tokens_used: Mapped[int] = mapped_column(Integer, nullable=False)
    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    tool_calls: Mapped[list[dict] | None] = mapped_column(JSONB, nullable=True)

    # Relationships
    fact: Mapped["Fact"] = relationship(back_populates="solution_details")

    __table_args__ = (
        UniqueConstraint("fact_id", name="uq_atomic_solution_fact"),
        Index("idx_atomic_solution_fact", "fact_id"),
        Index("idx_atomic_solution_agent", "agent_name"),
    )

    def __repr__(self) -> str:
        return f"<SolutionDetails(id={self.id}, agent={self.agent_name!r})>"


class PredictionDetails(Base):
    """
    Details for prediction facts (hypothesis tracking).

    Predictions have resolution criteria and outcome tracking for calibration.
    """

    __tablename__ = "atomic_prediction_details"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    fact_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("atomic_facts.id", ondelete="CASCADE"), nullable=False
    )

    # Resolution criteria
    resolution_type: Mapped[str | None] = mapped_column(String(50), nullable=True)
    resolution_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    resolution_event: Mapped[str | None] = mapped_column(Text, nullable=True)
    resolution_metric: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    # Verification
    verification_source: Mapped[str | None] = mapped_column(String(100), nullable=True)
    verification_url: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Outcome
    status: Mapped[str] = mapped_column(String(20), default="pending", nullable=False)
    outcome: Mapped[str | None] = mapped_column(String(20), nullable=True)
    outcome_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    actual_result: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Additional metadata
    tags: Mapped[list[str] | None] = mapped_column(ARRAY(String(50)), nullable=True)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Relationships
    fact: Mapped["Fact"] = relationship(back_populates="prediction_details")

    __table_args__ = (
        UniqueConstraint("fact_id", name="uq_atomic_prediction_fact"),
        Index("idx_atomic_prediction_fact", "fact_id"),
        Index("idx_atomic_prediction_status", "status"),
        Index("idx_atomic_prediction_resolution_date", "resolution_date"),
    )

    def __repr__(self) -> str:
        return f"<PredictionDetails(id={self.id}, status={self.status!r})>"


class FeedbackDetails(Base):
    """
    Details for feedback facts (explicit user signals on content).
    """

    __tablename__ = "atomic_feedback_details"

    # Keys
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    fact_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("atomic_facts.id", ondelete="CASCADE"), nullable=False
    )
    content_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("content.id"), nullable=True
    )

    # Core feedback
    signal: Mapped[str] = mapped_column(String(20), nullable=False)
    strength: Mapped[float] = mapped_column(Float, default=1.0, nullable=False)

    # Metadata (when and who)
    feedback_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    provider_type: Mapped[str | None] = mapped_column(String(50), nullable=True)
    provider: Mapped[str | None] = mapped_column(String(100), nullable=True)

    # Additional details
    tags: Mapped[list[str] | None] = mapped_column(ARRAY(String(50)), nullable=True)
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    context: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    # Relationships
    fact: Mapped["Fact"] = relationship(back_populates="feedback_details")

    __table_args__ = (
        UniqueConstraint("fact_id", name="uq_atomic_feedback_fact"),
        Index("idx_atomic_feedback_fact", "fact_id"),
        Index("idx_atomic_feedback_content", "content_id"),
        Index("idx_atomic_feedback_signal", "signal"),
        Index("idx_atomic_feedback_provider_type", "provider_type"),
        Index("idx_atomic_feedback_provider", "provider"),
        Index("idx_atomic_feedback_at", "feedback_at"),
    )

    def __repr__(self) -> str:
        return f"<FeedbackDetails(id={self.id}, signal={self.signal!r})>"


class DirectiveDetails(Base):
    """
    Details for directive facts (standing instructions/rules).
    """

    __tablename__ = "atomic_directive_details"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    fact_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("atomic_facts.id", ondelete="CASCADE"), nullable=False
    )

    directive_type: Mapped[str | None] = mapped_column(String(50), nullable=True)
    parsed_rules: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="active", nullable=False)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Relationships
    fact: Mapped["Fact"] = relationship(back_populates="directive_details")

    __table_args__ = (
        UniqueConstraint("fact_id", name="uq_atomic_directive_fact"),
        Index("idx_atomic_directive_fact", "fact_id"),
        Index("idx_atomic_directive_status", "status"),
    )

    def __repr__(self) -> str:
        return f"<DirectiveDetails(id={self.id}, type={self.directive_type!r})>"


class InteractionDetails(Base):
    """
    Details for interaction facts (implicit behavioral signals).
    """

    __tablename__ = "atomic_interaction_details"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    fact_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("atomic_facts.id", ondelete="CASCADE"), nullable=False
    )
    content_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("content.id"), nullable=True
    )

    interaction_type: Mapped[str] = mapped_column(String(50), nullable=False)
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    scroll_depth: Mapped[float | None] = mapped_column(Float, nullable=True)
    context: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    # Relationships
    fact: Mapped["Fact"] = relationship(back_populates="interaction_details")

    __table_args__ = (
        UniqueConstraint("fact_id", name="uq_atomic_interaction_fact"),
        Index("idx_atomic_interaction_fact", "fact_id"),
        Index("idx_atomic_interaction_content", "content_id"),
        Index("idx_atomic_interaction_type", "interaction_type"),
    )

    def __repr__(self) -> str:
        return f"<InteractionDetails(id={self.id}, type={self.interaction_type!r})>"


class PreferenceDetails(Base):
    """
    Details for preference facts (DPO training pairs).
    """

    __tablename__ = "atomic_preference_details"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    fact_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("atomic_facts.id", ondelete="CASCADE"), nullable=False
    )

    context: Mapped[str] = mapped_column(Text, nullable=False)
    chosen: Mapped[str] = mapped_column(Text, nullable=False)
    rejected: Mapped[str] = mapped_column(Text, nullable=False)
    margin: Mapped[float | None] = mapped_column(Float, nullable=True)
    metadata_: Mapped[dict | None] = mapped_column("metadata", JSONB, nullable=True)

    # Relationships
    fact: Mapped["Fact"] = relationship(back_populates="preference_details")

    __table_args__ = (
        UniqueConstraint("fact_id", name="uq_atomic_preference_fact"),
        Index("idx_atomic_preference_fact", "fact_id"),
    )

    def __repr__(self) -> str:
        return f"<PreferenceDetails(id={self.id})>"
