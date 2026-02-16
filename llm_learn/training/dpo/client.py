"""DPO training client with pair management.

Manages DPO training runs:
- Creating training runs
- Assigning preference pairs to runs (exclusive - each pair used once)
- Tracking run lifecycle (pending → running → completed/failed)
- Querying untrained pairs
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, cast

from appinfra.db.utils import detach
from appinfra.log import Logger
from sqlalchemy import (
    BigInteger,
    DateTime,
    ForeignKey,
    Index,
    String,
    Table,
    Text,
    UniqueConstraint,
    select,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Mapped, mapped_column

from llm_learn.core.base import Base, utc_now
from llm_learn.core.exceptions import ConflictError, NotFoundError, ValidationError
from llm_learn.memory.atomic.models import Fact, PreferenceDetails
from llm_learn.memory.isolation import build_context_filter

# Type alias for pair lists
PairList = list[tuple[Fact, PreferenceDetails]]


# =============================================================================
# ORM Models
# =============================================================================


class DpoRun(Base):
    """DPO training run metadata.

    Tracks the lifecycle of a DPO training run:
    - pending: Created, waiting in queue
    - running: Training in progress
    - completed: Successfully finished
    - failed: Terminated with error
    """

    __tablename__ = "dpo_runs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    context_key: Mapped[str | None] = mapped_column(String(255), nullable=True)
    adapter_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="pending", nullable=False)
    config: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    metrics: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        Index("idx_dpo_runs_context", "context_key"),
        Index("idx_dpo_runs_status", "status"),
        Index("idx_dpo_runs_adapter", "adapter_name"),
        Index("idx_dpo_runs_created", "created_at"),
        Index(
            "idx_dpo_runs_context_prefix",
            "context_key",
            postgresql_ops={"context_key": "varchar_pattern_ops"},
        ),
    )

    def __repr__(self) -> str:
        return f"<DpoRun(id={self.id}, status={self.status!r}, adapter={self.adapter_name!r})>"


class DpoRunPair(Base):
    """Junction table linking DPO runs to preference pairs.

    The UNIQUE constraint on preference_fact_id enforces exclusive assignment:
    each preference pair can only be assigned to ONE training run at a time.
    """

    __tablename__ = "dpo_run_pairs"

    run_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("dpo_runs.id", ondelete="CASCADE"), primary_key=True
    )
    preference_fact_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("atomic_facts.id", ondelete="CASCADE"), primary_key=True
    )
    assigned_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )

    __table_args__ = (
        Index("idx_dpo_run_pairs_run", "run_id"),
        Index("idx_dpo_run_pairs_fact", "preference_fact_id"),
        UniqueConstraint("preference_fact_id", name="uq_dpo_run_pair_exclusive"),
    )

    def __repr__(self) -> str:
        return f"<DpoRunPair(run={self.run_id}, fact={self.preference_fact_id})>"


# =============================================================================
# Dataclass for API returns (session-independent)
# =============================================================================


@dataclass
class DpoRunInfo:
    """DPO run information, detached from database session."""

    id: int
    context_key: str | None
    adapter_name: str | None
    status: str
    config: dict | None
    metrics: dict | None
    error_message: str | None
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None

    @classmethod
    def from_model(cls, run: DpoRun) -> DpoRunInfo:
        """Create from ORM model."""
        return cls(
            id=run.id,
            context_key=run.context_key,
            adapter_name=run.adapter_name,
            status=run.status,
            config=run.config,
            metrics=run.metrics,
            error_message=run.error_message,
            created_at=run.created_at,
            started_at=run.started_at,
            completed_at=run.completed_at,
        )


# =============================================================================
# Client
# =============================================================================

# Valid status values and allowed transitions
VALID_STATUSES = {"pending", "running", "completed", "failed", "deleted"}
STATUS_TRANSITIONS = {
    "pending": {"running", "failed", "deleted"},  # Can start, fail, or be deleted
    "running": {
        "pending",
        "completed",
        "failed",
        "deleted",
    },  # Can reset, complete, fail, or delete
    "completed": {"deleted"},  # Can only be deleted
    "failed": {"deleted"},  # Can only be deleted
    "deleted": set(),  # Terminal state
}


class DpoClient:
    """Client for DPO training with pair management.

    Manages:
    - Creating/listing/deleting DPO training runs
    - Assigning preference pairs to runs (exclusive - each pair used once)
    - Tracking run lifecycle (pending → running → completed/failed)
    - Querying untrained pairs
    """

    def __init__(
        self,
        lg: Logger,
        session_factory: Callable[[], Any],
        context_key: str | None,
        ensure_schema: bool = False,
    ) -> None:
        """
        Initialize DPO client.

        Args:
            lg: Logger instance.
            session_factory: Callable that returns a context manager for database sessions.
            context_key: Context key to scope operations (None = no filtering).
            ensure_schema: If True, create DPO tables if they don't exist.
        """
        self._lg = lg
        self._session_factory = session_factory
        self.context_key = context_key

        if ensure_schema:
            self._ensure_tables()

    def _ensure_tables(self) -> None:
        """Create DPO tables if they don't exist."""
        with self._session_factory() as session:
            engine = session.get_bind()
            tables = [
                cast(Table, DpoRun.__table__),
                cast(Table, DpoRunPair.__table__),
            ]
            Base.metadata.create_all(engine, tables=tables)
            self._lg.debug("ensured DPO tables exist")

    def _build_context_filter(self, column):
        """Build context filter condition with pattern matching support."""
        return build_context_filter(self.context_key, column)

    def _get_run(self, session, run_id: int) -> DpoRun | None:
        """Get run by ID, verifying context ownership."""
        stmt = select(DpoRun).where(DpoRun.id == run_id)
        context_filter = self._build_context_filter(DpoRun.context_key)
        if context_filter is not None:
            stmt = stmt.where(context_filter)
        return cast(DpoRun | None, session.scalar(stmt))

    # -------------------------------------------------------------------------
    # CRUD operations
    # -------------------------------------------------------------------------

    def create(
        self,
        adapter_name: str | None = None,
        config: dict | None = None,
    ) -> DpoRunInfo:
        """
        Create a new DPO training run.

        Args:
            adapter_name: Optional name for the adapter being trained.
            config: Optional training configuration dict.

        Returns:
            DpoRunInfo with the new run's details.

        Raises:
            ValidationError: If client was created with a glob-pattern context_key.
        """
        if _is_pattern(self.context_key):
            raise ValidationError(
                "Cannot create runs with a glob-pattern context_key; use an exact context."
            )
        run = DpoRun(
            context_key=self.context_key,
            adapter_name=adapter_name,
            config=config,
        )

        with self._session_factory() as session:
            session.add(run)
            session.flush()
            info = DpoRunInfo.from_model(run)
            self._lg.debug(
                "created DPO run",
                extra={"run_id": info.id, "adapter_name": adapter_name},
            )
            return info

    def get(self, run_id: int) -> DpoRunInfo | None:
        """Get a DPO run by ID."""
        with self._session_factory() as session:
            run = self._get_run(session, run_id)
            if run is None:
                return None
            return DpoRunInfo.from_model(run)

    def list(
        self,
        status: str | None = None,
        limit: int = 100,
        descending: bool = True,
        include_deleted: bool = False,
    ) -> list[DpoRunInfo]:
        """
        List DPO runs.

        Args:
            status: Filter by status (pending, running, completed, failed, deleted).
            limit: Maximum number of runs to return.
            descending: Order by created_at descending (newest first).
            include_deleted: Include deleted runs (default False).

        Returns:
            List of DpoRunInfo.
        """
        if status is not None and status not in VALID_STATUSES:
            raise ValidationError(f"status must be one of {VALID_STATUSES}")

        with self._session_factory() as session:
            stmt = select(DpoRun)
            context_filter = self._build_context_filter(DpoRun.context_key)
            if context_filter is not None:
                stmt = stmt.where(context_filter)
            if status is not None:
                stmt = stmt.where(DpoRun.status == status)
            elif not include_deleted:
                stmt = stmt.where(DpoRun.status != "deleted")
            order = DpoRun.created_at.desc() if descending else DpoRun.created_at.asc()
            stmt = stmt.order_by(order).limit(limit)

            runs = list(session.scalars(stmt).all())
            return [DpoRunInfo.from_model(r) for r in runs]

    def delete(self, run_id: int) -> bool:
        """Delete a DPO run (frees assigned pairs for reuse)."""
        with self._session_factory() as session:
            run = self._get_run(session, run_id)
            if run is None:
                return False
            session.delete(run)
            self._lg.debug("deleted DPO run", extra={"run_id": run_id})
            return True

    # -------------------------------------------------------------------------
    # Pair assignment
    # -------------------------------------------------------------------------

    def assign_pairs(self, run_id: int, pair_fact_ids: Sequence[int]) -> int:
        """
        Assign preference pairs to a DPO run.

        Each pair can only be assigned to ONE run (exclusive assignment).
        Attempting to assign an already-assigned pair raises ConflictError.

        Args:
            run_id: The run ID to assign pairs to.
            pair_fact_ids: List of preference fact IDs to assign.

        Returns:
            Number of pairs assigned.

        Raises:
            NotFoundError: If run_id doesn't exist or isn't accessible.
            ConflictError: If any pair is already assigned to another run.
        """
        if not pair_fact_ids:
            return 0

        with self._session_factory() as session:
            run = self._get_run(session, run_id)
            if run is None:
                raise NotFoundError(f"DPO run {run_id} not found")

            self._check_pairs_available(session, pair_fact_ids)
            self._insert_pairs(session, run_id, pair_fact_ids)

            self._lg.debug(
                "assigned pairs to DPO run",
                extra={"run_id": run_id, "count": len(pair_fact_ids)},
            )
            return len(pair_fact_ids)

    def _check_pairs_available(self, session, pair_fact_ids: Sequence[int]) -> None:
        """Check that none of the pairs are already assigned."""
        existing_stmt = select(DpoRunPair.preference_fact_id).where(
            DpoRunPair.preference_fact_id.in_(pair_fact_ids)
        )
        existing_ids = set(session.scalars(existing_stmt).all())
        if existing_ids:
            raise ConflictError(f"Pairs already assigned to other runs: {sorted(existing_ids)}")

    def _insert_pairs(self, session, run_id: int, pair_fact_ids: Sequence[int]) -> None:
        """Insert pairs, handling concurrent assignment race condition."""
        try:
            for fact_id in pair_fact_ids:
                session.add(DpoRunPair(run_id=run_id, preference_fact_id=fact_id))
            session.flush()
        except IntegrityError as e:
            session.rollback()
            if "uq_dpo_run_pair_exclusive" in str(e):
                raise ConflictError(
                    "Pairs were assigned by another process during operation"
                ) from e
            raise

    def get_pairs(self, run_id: int) -> PairList:
        """
        Get all preference pairs assigned to a DPO run.

        Args:
            run_id: The run ID.

        Returns:
            List of (Fact, PreferenceDetails) tuples.

        Raises:
            NotFoundError: If run_id doesn't exist or isn't accessible.
        """
        with self._session_factory() as session:
            run = self._get_run(session, run_id)
            if run is None:
                raise NotFoundError(f"DPO run {run_id} not found")

            stmt = (
                select(Fact, PreferenceDetails)
                .join(PreferenceDetails, Fact.id == PreferenceDetails.fact_id)
                .join(DpoRunPair, DpoRunPair.preference_fact_id == Fact.id)
                .where(DpoRunPair.run_id == run_id, Fact.type == "preference")
                .order_by(Fact.created_at)
            )

            results = list(session.execute(stmt).all())
            return self._detach_pairs(results, session)

    def get_untrained_pairs(
        self,
        *,
        min_margin: float | None = None,
        limit: int | None = None,
    ) -> PairList:
        """
        Get preference pairs that haven't been assigned to any DPO run.

        Args:
            min_margin: Only include pairs with margin >= this value.
            limit: Maximum number of pairs to return.

        Returns:
            List of (Fact, PreferenceDetails) tuples for untrained pairs.
        """
        with self._session_factory() as session:
            stmt = self._build_untrained_pairs_query(min_margin, limit)
            results = list(session.execute(stmt).all())
            return self._detach_pairs(results, session)

    def _build_untrained_pairs_query(self, min_margin: float | None, limit: int | None):
        """Build query for untrained preference pairs."""
        assigned_subq = select(DpoRunPair.preference_fact_id)
        stmt = (
            select(Fact, PreferenceDetails)
            .join(PreferenceDetails, Fact.id == PreferenceDetails.fact_id)
            .where(
                Fact.type == "preference",
                Fact.active == True,  # noqa: E712
                ~Fact.id.in_(assigned_subq),
            )
        )
        context_filter = build_context_filter(self.context_key, Fact.context_key)
        if context_filter is not None:
            stmt = stmt.where(context_filter)
        if min_margin is not None:
            stmt = stmt.where(PreferenceDetails.margin >= min_margin)
        stmt = stmt.order_by(Fact.created_at)
        if limit is not None:
            stmt = stmt.limit(limit)
        return stmt

    def _detach_pairs(self, results, session) -> PairList:
        """Detach fact/details pairs from session for safe return."""
        detached = []
        for fact, details in results:
            detach(fact, session)
            detach(details, session)
            detached.append((fact, details))
        return detached

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    def start(self, run_id: int) -> None:
        """Mark a DPO run as started."""
        self._transition(run_id, "running")

    def complete(self, run_id: int, metrics: dict | None = None) -> None:
        """Mark a DPO run as completed."""
        self._transition(run_id, "completed", metrics=metrics)

    def fail(self, run_id: int, error: str) -> None:
        """Mark a DPO run as failed."""
        self._transition(run_id, "failed", error_message=error)

    def reset(self, run_id: int) -> None:
        """Reset a running DPO run back to pending (for retry after transient failure)."""
        self._transition(run_id, "pending", clear_started=True)

    def _transition(
        self,
        run_id: int,
        new_status: str,
        metrics: dict | None = None,
        error_message: str | None = None,
        clear_started: bool = False,
    ) -> None:
        """Transition a run to a new status with validation."""
        with self._session_factory() as session:
            run = self._get_run(session, run_id)
            if run is None:
                raise NotFoundError(f"DPO run {run_id} not found")

            allowed = STATUS_TRANSITIONS.get(run.status, set())
            if new_status not in allowed:
                raise ValidationError(
                    f"Cannot transition from '{run.status}' to '{new_status}'. "
                    f"Allowed transitions: {allowed or 'none (terminal state)'}"
                )

            self._apply_transition(run, new_status, metrics, error_message, clear_started)
            self._lg.debug(
                "DPO run status changed", extra={"run_id": run_id, "new_status": new_status}
            )

    def _apply_transition(self, run, new_status, metrics, error_message, clear_started):
        """Apply status transition and update timestamps/fields."""
        run.status = new_status
        now = utc_now()
        if new_status == "running":
            run.started_at = now
        elif new_status in ("completed", "failed"):
            run.completed_at = now
        elif clear_started:
            run.started_at = None
        if metrics is not None:
            run.metrics = metrics
        if error_message is not None:
            run.error_message = error_message

    # -------------------------------------------------------------------------
    # Utility
    # -------------------------------------------------------------------------

    def reset_all(self) -> int:
        """Delete all DPO runs for this context (frees all pairs)."""
        with self._session_factory() as session:
            stmt = select(DpoRun)
            context_filter = self._build_context_filter(DpoRun.context_key)
            if context_filter is not None:
                stmt = stmt.where(context_filter)

            runs = list(session.scalars(stmt).all())
            count = len(runs)

            for run in runs:
                session.delete(run)

            self._lg.info("reset all DPO runs", extra={"count": count})
            return count


def _is_pattern(context_key: str | None) -> bool:
    """Check if context_key contains glob pattern characters."""
    if context_key is None:
        return False
    return "*" in context_key or "?" in context_key
