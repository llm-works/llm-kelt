"""Shared training models and utilities.

Contains ORM models and helper functions used by all training method clients
(DPO, SFT, etc.). Centralizes training run management to avoid duplication.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, cast

from sqlalchemy import BigInteger, DateTime, ForeignKey, Index, String, Text, or_, select
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from llm_learn.core.base import Base, utc_now
from llm_learn.memory.isolation import build_context_filter

if TYPE_CHECKING:
    from sqlalchemy.orm import Session


# =============================================================================
# Helper Functions
# =============================================================================


def _not_deleted_filter(model):
    """Build filter for non-deleted runs (system_status is NULL or deleted != true)."""
    return or_(
        model.system_status.is_(None),
        model.system_status["deleted"].astext != "true",
    )


def _is_pattern(context_key: str | None) -> bool:
    """Check if context_key contains glob pattern characters."""
    if context_key is None:
        return False
    return "*" in context_key or "?" in context_key


# =============================================================================
# ORM Models
# =============================================================================


class Run(Base):
    """Generic training run metadata.

    Supports multiple training methods (DPO, SFT, RLHF).
    Tracks the lifecycle of a training run:
    - pending: Created, waiting in queue
    - running: Training in progress
    - completed: Successfully finished
    - failed: Terminated with error

    System state tracked via system_status JSONB (e.g., {"deleted": true, "deleted_at": "..."}).
    """

    __tablename__ = "training_runs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    method: Mapped[str] = mapped_column(String(20), nullable=False)
    context_key: Mapped[str | None] = mapped_column(String(255), nullable=True)
    adapter: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    based_on: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("training_runs.id", ondelete="SET NULL"), nullable=True
    )
    status: Mapped[str] = mapped_column(String(20), default="pending", nullable=False)
    config: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    metrics: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    system_status: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        Index("idx_training_runs_method", "method"),
        Index("idx_training_runs_context", "context_key"),
        Index("idx_training_runs_status", "status"),
        Index("idx_training_runs_created", "created_at"),
        Index("idx_training_runs_based_on", "based_on"),
        Index(
            "idx_training_runs_context_prefix",
            "context_key",
            postgresql_ops={"context_key": "varchar_pattern_ops"},
        ),
    )

    @property
    def adapter_name(self) -> str | None:
        """Get adapter name from adapter JSONB."""
        if self.adapter is None:
            return None
        return self.adapter.get("name")

    @property
    def is_deleted(self) -> bool:
        """Check if run is soft-deleted."""
        if self.system_status is None:
            return False
        return bool(self.system_status.get("deleted", False))

    def __repr__(self) -> str:
        return f"<Run(id={self.id}, method={self.method!r}, status={self.status!r})>"


# =============================================================================
# Dataclass for API returns (session-independent)
# =============================================================================


@dataclass
class RunInfo:
    """Training run information, detached from database session."""

    id: int
    method: str
    context_key: str | None
    adapter: dict | None
    based_on: int | None
    status: str
    config: dict | None
    metrics: dict | None
    error_message: str | None
    system_status: dict | None
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None

    @property
    def adapter_name(self) -> str | None:
        """Get adapter name from adapter dict."""
        if self.adapter is None:
            return None
        return self.adapter.get("name")

    @property
    def is_deleted(self) -> bool:
        """Check if run is soft-deleted."""
        if self.system_status is None:
            return False
        return bool(self.system_status.get("deleted", False))

    @classmethod
    def from_model(cls, run: Run) -> RunInfo:
        """Create from ORM model."""
        return cls(
            id=run.id,
            method=run.method,
            context_key=run.context_key,
            adapter=run.adapter,
            based_on=run.based_on,
            status=run.status,
            config=run.config,
            metrics=run.metrics,
            error_message=run.error_message,
            system_status=run.system_status,
            created_at=run.created_at,
            started_at=run.started_at,
            completed_at=run.completed_at,
        )


# =============================================================================
# Status Constants
# =============================================================================

VALID_STATUSES = {"pending", "running", "completed", "failed", "cancelled"}
STATUS_TRANSITIONS = {
    "pending": {"running", "failed", "cancelled"},
    "running": {"pending", "completed", "failed"},
    "completed": set(),  # Terminal - use soft-delete to hide
    "failed": set(),  # Terminal - use soft-delete to hide
    "cancelled": set(),  # Terminal - user cancelled before execution
}


# =============================================================================
# Cross-Method Lineage
# =============================================================================


def get_parent_run(session: Session, run_id: int, context_key: str | None) -> Run | None:
    """Get any run by ID for lineage (cross-method allowed).

    Unlike method-specific _get_run(), this doesn't filter by training method,
    enabling cross-method lineage chains like SFT→DPO.

    Args:
        session: Database session.
        run_id: ID of the parent run to look up.
        context_key: Context key for isolation filtering.

    Returns:
        Run or None if not found.
    """
    stmt = select(Run).where(
        Run.id == run_id,
        _not_deleted_filter(Run),
    )
    context_filter = build_context_filter(context_key, Run.context_key)
    if context_filter is not None:
        stmt = stmt.where(context_filter)
    return cast(Run | None, session.scalar(stmt))
