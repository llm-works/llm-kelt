"""Training run tracking for iterative DPO workflows.

Tracks which preference pairs have been used in which training runs to:
- Prevent duplicate training on same pairs (biases model)
- Support incremental training on new pairs only
- Allow reset without losing base data (jokes, ratings, pairs)
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, cast

from appinfra.db.utils import detach
from appinfra.log import Logger
from sqlalchemy import BigInteger, DateTime, ForeignKey, Index, String, Text, select
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from llm_learn.core.base import Base, utc_now
from llm_learn.core.exceptions import ConflictError, NotFoundError, ValidationError
from llm_learn.memory.atomic.models import Fact, PreferenceDetails
from llm_learn.memory.isolation import build_context_filter

# Type alias to avoid shadowing by method named 'list'
PairList = list[tuple[Fact, PreferenceDetails]]


# =============================================================================
# ORM Models
# =============================================================================


class TrainingRun(Base):
    """Training run metadata.

    Tracks the lifecycle of a training run:
    - pending: Created but not started
    - running: Training in progress
    - completed: Successfully finished
    - failed: Terminated with error
    """

    __tablename__ = "training_runs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    context_key: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
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

    # Relationship to assigned pairs
    pairs: Mapped[list[TrainingRunPair]] = relationship(
        back_populates="training_run", passive_deletes=True
    )

    __table_args__ = (
        Index("idx_training_runs_context", "context_key"),
        Index("idx_training_runs_status", "status"),
        Index("idx_training_runs_adapter", "adapter_name"),
        Index("idx_training_runs_created", "created_at"),
        Index(
            "idx_training_runs_context_prefix",
            "context_key",
            postgresql_ops={"context_key": "varchar_pattern_ops"},
        ),
    )

    def __repr__(self) -> str:
        return f"<TrainingRun(id={self.id}, status={self.status!r}, adapter={self.adapter_name!r})>"


class TrainingRunPair(Base):
    """Junction table linking training runs to preference pairs.

    The UNIQUE constraint on preference_fact_id enforces exclusive assignment:
    each preference pair can only be assigned to ONE training run at a time.
    """

    __tablename__ = "training_run_pairs"

    training_run_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("training_runs.id", ondelete="CASCADE"), primary_key=True
    )
    preference_fact_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("atomic_facts.id", ondelete="CASCADE"), primary_key=True
    )
    assigned_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )

    # Relationships
    training_run: Mapped[TrainingRun] = relationship(back_populates="pairs")

    __table_args__ = (
        Index("idx_training_run_pairs_run", "training_run_id"),
        Index("idx_training_run_pairs_fact", "preference_fact_id"),
    )

    def __repr__(self) -> str:
        return f"<TrainingRunPair(run={self.training_run_id}, fact={self.preference_fact_id})>"


# =============================================================================
# Dataclass for API returns (session-independent)
# =============================================================================


@dataclass
class TrainingRunInfo:
    """Training run information, detached from database session.

    Use this for returning run info from client methods.
    """

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
    def from_model(cls, run: TrainingRun) -> TrainingRunInfo:
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
VALID_STATUSES = {"pending", "running", "completed", "failed"}
STATUS_TRANSITIONS = {
    "pending": {"running", "failed"},  # Can start or fail directly
    "running": {"completed", "failed"},  # Can complete or fail
    "completed": set(),  # Terminal state
    "failed": set(),  # Terminal state
}


class TrainingRunClient:
    """Client for managing training runs.

    Tracks which preference pairs have been used in which training runs.
    Scoped to a context_key for isolation.
    """

    def __init__(
        self,
        lg: Logger,
        session_factory: Callable[[], Any],
        context_key: str | None,
        ensure_schema: bool = False,
    ) -> None:
        """
        Initialize client scoped to a specific context.

        Args:
            lg: Logger instance for all client operations.
            session_factory: Callable that returns a context manager for database sessions.
            context_key: Context key to scope all operations to (None = no filtering).
                Supports SQL LIKE patterns (% and _) for prefix/pattern matching.
            ensure_schema: If True, create training tables if they don't exist.
        """
        self._lg = lg
        self._session_factory = session_factory
        self.context_key = context_key

        if ensure_schema:
            self._ensure_tables()

    def _ensure_tables(self) -> None:
        """Create training tables if they don't exist."""
        from sqlalchemy import Table

        with self._session_factory() as session:
            engine = session.get_bind()
            tables = [
                cast(Table, TrainingRun.__table__),
                cast(Table, TrainingRunPair.__table__),
            ]
            Base.metadata.create_all(engine, tables=tables)
            self._lg.debug("ensured training tables exist")

    def _build_context_filter(self, column):
        """Build context filter condition with pattern matching support."""
        return build_context_filter(self.context_key, column)

    def _get_run(self, session, run_id: int) -> TrainingRun | None:
        """Get run by ID, verifying context ownership."""
        stmt = select(TrainingRun).where(TrainingRun.id == run_id)
        context_filter = self._build_context_filter(TrainingRun.context_key)
        if context_filter is not None:
            stmt = stmt.where(context_filter)
        return cast(TrainingRun | None, session.scalar(stmt))

    # -------------------------------------------------------------------------
    # CRUD operations
    # -------------------------------------------------------------------------

    def create(
        self,
        adapter_name: str | None = None,
        config: dict | None = None,
    ) -> TrainingRunInfo:
        """
        Create a new training run.

        Args:
            adapter_name: Optional name for the adapter being trained.
            config: Optional training configuration dict.

        Returns:
            TrainingRunInfo with the new run's details.
        """
        run = TrainingRun(
            context_key=self.context_key if not _is_pattern(self.context_key) else None,
            adapter_name=adapter_name,
            config=config,
        )

        with self._session_factory() as session:
            session.add(run)
            session.flush()  # Get ID
            info = TrainingRunInfo.from_model(run)
            self._lg.debug(
                "created training run",
                extra={"run_id": info.id, "adapter_name": adapter_name},
            )
            return info

    def get(self, run_id: int) -> TrainingRunInfo | None:
        """
        Get a training run by ID.

        Args:
            run_id: The run ID.

        Returns:
            TrainingRunInfo or None if not found.
        """
        with self._session_factory() as session:
            run = self._get_run(session, run_id)
            if run is None:
                return None
            return TrainingRunInfo.from_model(run)

    def list(
        self,
        status: str | None = None,
        limit: int = 100,
        descending: bool = True,
    ) -> list[TrainingRunInfo]:
        """
        List training runs.

        Args:
            status: Filter by status (pending, running, completed, failed).
            limit: Maximum number of runs to return.
            descending: Order by created_at descending (newest first).

        Returns:
            List of TrainingRunInfo.
        """
        if status is not None and status not in VALID_STATUSES:
            raise ValidationError(f"status must be one of {VALID_STATUSES}")

        with self._session_factory() as session:
            stmt = select(TrainingRun)
            context_filter = self._build_context_filter(TrainingRun.context_key)
            if context_filter is not None:
                stmt = stmt.where(context_filter)
            if status is not None:
                stmt = stmt.where(TrainingRun.status == status)
            order = TrainingRun.created_at.desc() if descending else TrainingRun.created_at.asc()
            stmt = stmt.order_by(order).limit(limit)

            runs = list(session.scalars(stmt).all())
            return [TrainingRunInfo.from_model(r) for r in runs]

    def delete(self, run_id: int) -> bool:
        """
        Delete a training run.

        When a run is deleted, its pair assignments are also deleted (CASCADE),
        freeing those pairs for use in future runs.

        Args:
            run_id: The run ID to delete.

        Returns:
            True if deleted, False if not found.
        """
        with self._session_factory() as session:
            run = self._get_run(session, run_id)
            if run is None:
                return False
            session.delete(run)
            self._lg.debug("deleted training run", extra={"run_id": run_id})
            return True

    # -------------------------------------------------------------------------
    # Pair assignment
    # -------------------------------------------------------------------------

    def assign_pairs(self, run_id: int, pair_fact_ids: Sequence[int]) -> int:
        """
        Assign preference pairs to a training run.

        Each pair can only be assigned to ONE run (exclusive assignment).
        Attempting to assign an already-assigned pair will raise ConflictError.

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
                raise NotFoundError(f"Training run {run_id} not found")

            # Check for already-assigned pairs
            existing_stmt = select(TrainingRunPair.preference_fact_id).where(
                TrainingRunPair.preference_fact_id.in_(pair_fact_ids)
            )
            existing_ids = set(session.scalars(existing_stmt).all())

            if existing_ids:
                raise ConflictError(f"Pairs already assigned to other runs: {sorted(existing_ids)}")

            # Assign pairs
            for fact_id in pair_fact_ids:
                pair = TrainingRunPair(training_run_id=run_id, preference_fact_id=fact_id)
                session.add(pair)

            session.flush()
            self._lg.debug(
                "assigned pairs to training run",
                extra={"run_id": run_id, "count": len(pair_fact_ids)},
            )
            return len(pair_fact_ids)

    def _detach_pairs(self, results, session) -> PairList:
        """Detach fact/details pairs from session for safe return."""
        detached = []
        for fact, details in results:
            detach(fact, session)
            detach(details, session)
            detached.append((fact, details))
        return detached

    def get_pairs(self, run_id: int) -> PairList:
        """
        Get all preference pairs assigned to a training run.

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
                raise NotFoundError(f"Training run {run_id} not found")

            stmt = (
                select(Fact, PreferenceDetails)
                .join(PreferenceDetails, Fact.id == PreferenceDetails.fact_id)
                .join(TrainingRunPair, TrainingRunPair.preference_fact_id == Fact.id)
                .where(TrainingRunPair.training_run_id == run_id, Fact.type == "preference")
                .order_by(Fact.created_at)
            )

            results = list(session.execute(stmt).all())
            return self._detach_pairs(results, session)

    def _build_untrained_pairs_query(self, min_margin: float | None, limit: int | None):
        """Build query for untrained preference pairs."""
        assigned_subq = select(TrainingRunPair.preference_fact_id)
        stmt = (
            select(Fact, PreferenceDetails)
            .join(PreferenceDetails, Fact.id == PreferenceDetails.fact_id)
            .where(Fact.type == "preference", Fact.active == True, ~Fact.id.in_(assigned_subq))  # noqa: E712
        )
        context_filter = self._build_context_filter(Fact.context_key)
        if context_filter is not None:
            stmt = stmt.where(context_filter)
        if min_margin is not None:
            stmt = stmt.where(PreferenceDetails.margin >= min_margin)
        stmt = stmt.order_by(Fact.created_at)
        if limit is not None:
            stmt = stmt.limit(limit)
        return stmt

    def get_untrained_pairs(
        self,
        *,
        min_margin: float | None = None,
        limit: int | None = None,
    ) -> PairList:
        """
        Get preference pairs that haven't been assigned to any training run.

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

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    def start(self, run_id: int) -> None:
        """
        Mark a training run as started.

        Args:
            run_id: The run ID to start.

        Raises:
            NotFoundError: If run_id doesn't exist.
            ValidationError: If run is not in 'pending' status.
        """
        self._transition(run_id, "running")

    def complete(self, run_id: int, metrics: dict | None = None) -> None:
        """
        Mark a training run as completed.

        Args:
            run_id: The run ID.
            metrics: Optional metrics dict to store.

        Raises:
            NotFoundError: If run_id doesn't exist.
            ValidationError: If run is not in 'running' status.
        """
        self._transition(run_id, "completed", metrics=metrics)

    def fail(self, run_id: int, error: str) -> None:
        """
        Mark a training run as failed.

        Args:
            run_id: The run ID.
            error: Error message describing the failure.

        Raises:
            NotFoundError: If run_id doesn't exist.
            ValidationError: If run is in a terminal status.
        """
        self._transition(run_id, "failed", error_message=error)

    def _transition(
        self,
        run_id: int,
        new_status: str,
        metrics: dict | None = None,
        error_message: str | None = None,
    ) -> None:
        """Transition a run to a new status with validation."""
        with self._session_factory() as session:
            run = self._get_run(session, run_id)
            if run is None:
                raise NotFoundError(f"Training run {run_id} not found")

            allowed = STATUS_TRANSITIONS.get(run.status, set())
            if new_status not in allowed:
                raise ValidationError(
                    f"Cannot transition from '{run.status}' to '{new_status}'. "
                    f"Allowed transitions: {allowed or 'none (terminal state)'}"
                )

            now = utc_now()
            run.status = new_status

            if new_status == "running":
                run.started_at = now
            elif new_status in ("completed", "failed"):
                run.completed_at = now

            if metrics is not None:
                run.metrics = metrics
            if error_message is not None:
                run.error_message = error_message

            self._lg.debug(
                "training run status changed",
                extra={"run_id": run_id, "new_status": new_status},
            )

    # -------------------------------------------------------------------------
    # Utility
    # -------------------------------------------------------------------------

    def reset_all(self) -> int:
        """
        Delete all training runs for this context.

        This frees all preference pairs for reuse in new training runs.

        Returns:
            Number of runs deleted.
        """
        with self._session_factory() as session:
            stmt = select(TrainingRun)
            context_filter = self._build_context_filter(TrainingRun.context_key)
            if context_filter is not None:
                stmt = stmt.where(context_filter)

            runs = list(session.scalars(stmt).all())
            count = len(runs)

            for run in runs:
                session.delete(run)

            self._lg.info("reset all training runs", extra={"count": count})
            return count


def _is_pattern(context_key: str | None) -> bool:
    """Check if context_key contains glob pattern characters."""
    if context_key is None:
        return False
    return "*" in context_key or "?" in context_key
