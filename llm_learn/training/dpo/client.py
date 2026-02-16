"""DPO training client with pair management.

Manages DPO training runs:
- Creating training runs (with lineage via based_on)
- Assigning preference pairs to runs (lineage-aware - pairs excluded from ancestor runs)
- Tracking run lifecycle (pending → running → completed/failed)
- Moving pairs from pending to trained on completion
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, cast

from appinfra.log import Logger
from sqlalchemy import (
    BigInteger,
    DateTime,
    ForeignKey,
    Index,
    String,
    Table,
    Text,
    or_,
    select,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from llm_learn.core.base import Base, utc_now
from llm_learn.core.exceptions import NotFoundError, ValidationError
from llm_learn.memory.isolation import build_context_filter

from .export import PairTuple


def _not_deleted_filter(model):
    """Build filter for non-deleted runs (system_status is NULL or deleted != true)."""
    return or_(
        model.system_status.is_(None),
        model.system_status["deleted"].astext != "true",
    )


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


class PendingPair(Base):
    """Temporary table for pairs assigned to pending/running DPO runs.

    Pairs reference solution facts via chosen/rejected fact IDs.
    Deleted when run completes (moved to trained) or fails (freed for retry).
    """

    __tablename__ = "dpo_pending_pairs"

    run_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("training_runs.id", ondelete="CASCADE"), primary_key=True
    )
    chosen_fact_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("atomic_facts.id", ondelete="CASCADE"), primary_key=True
    )
    rejected_fact_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("atomic_facts.id", ondelete="CASCADE"), primary_key=True
    )
    prompt: Mapped[str] = mapped_column(Text, nullable=False)
    assigned_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )

    __table_args__ = (Index("idx_dpo_pending_pairs_run", "run_id"),)

    def __repr__(self) -> str:
        return f"<PendingPair(run={self.run_id}, chosen={self.chosen_fact_id}, rejected={self.rejected_fact_id})>"


class TrainedPair(Base):
    """Permanent history of pairs used in completed DPO training.

    Pairs are moved here when a run completes successfully.
    Used for lineage-based exclusion in future runs.
    """

    __tablename__ = "dpo_trained_pairs"

    run_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("training_runs.id", ondelete="CASCADE"), primary_key=True
    )
    chosen_fact_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("atomic_facts.id", ondelete="CASCADE"), primary_key=True
    )
    rejected_fact_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("atomic_facts.id", ondelete="CASCADE"), primary_key=True
    )
    prompt: Mapped[str] = mapped_column(Text, nullable=False)
    trained_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )

    __table_args__ = (Index("idx_dpo_trained_pairs_run", "run_id"),)

    def __repr__(self) -> str:
        return f"<TrainedPair(run={self.run_id}, chosen={self.chosen_fact_id}, rejected={self.rejected_fact_id})>"


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
# Client
# =============================================================================

# Valid status values and allowed transitions (deleted handled via system_status)
VALID_STATUSES = {"pending", "running", "completed", "failed"}
STATUS_TRANSITIONS = {
    "pending": {"running", "failed"},
    "running": {"pending", "completed", "failed"},
    "completed": set(),  # Terminal - use soft-delete
    "failed": set(),  # Terminal - use soft-delete
}


class Client:
    """Client for DPO training with pair management.

    Manages:
    - Creating/listing/deleting DPO training runs
    - Assigning preference pairs to runs (lineage-aware)
    - Tracking run lifecycle (pending → running → completed/failed)
    - Moving pairs from pending to trained on completion
    """

    def __init__(
        self,
        lg: Logger,
        session_factory: Callable[[], Any],
        context_key: str | None,
        ensure_schema: bool = False,
    ) -> None:
        """Initialize DPO client.

        Args:
            lg: Logger instance.
            session_factory: Callable that returns a context manager for database sessions.
            context_key: Context key to scope operations (None = no filtering).
            ensure_schema: If True, create training tables if they don't exist.
        """
        self._lg = lg
        self._session_factory = session_factory
        self.context_key = context_key
        self.method = "dpo"

        if ensure_schema:
            self._ensure_tables()

    def _ensure_tables(self) -> None:
        """Create training tables if they don't exist."""
        with self._session_factory() as session:
            engine = session.get_bind()
            tables = [
                cast(Table, Run.__table__),
                cast(Table, PendingPair.__table__),
                cast(Table, TrainedPair.__table__),
            ]
            Base.metadata.create_all(engine, tables=tables)
            self._lg.debug("ensured training tables exist")

    def _build_context_filter(self, column):
        """Build context filter condition with pattern matching support."""
        return build_context_filter(self.context_key, column)

    def _get_run(self, session, run_id: int) -> Run | None:
        """Get run by ID, verifying context ownership and method."""
        stmt = select(Run).where(
            Run.id == run_id,
            Run.method == self.method,
            _not_deleted_filter(Run),
        )
        context_filter = self._build_context_filter(Run.context_key)
        if context_filter is not None:
            stmt = stmt.where(context_filter)
        return cast(Run | None, session.scalar(stmt))

    # -------------------------------------------------------------------------
    # CRUD operations
    # -------------------------------------------------------------------------

    def create(
        self,
        adapter_name: str | None = None,
        config: dict | None = None,
        based_on: int | None = None,
        replace_stale: bool = True,
        on_replace: Callable[[RunInfo], bool] | None = None,
    ) -> RunInfo:
        """Create a new DPO training run, or reset an existing pending run.

        If a pending run already exists for this context and `replace_stale=True`,
        clears its pairs and returns it (no new run created). This allows callers
        to safely retry run creation without accumulating orphan runs.

        Args:
            adapter_name: Optional name for the adapter being trained.
            config: Optional training configuration dict.
            based_on: Optional run ID to base this run on (for lineage).
            replace_stale: If True (default), reuse any existing pending run for
                this context by clearing its pairs. If False, always create new.
            on_replace: Optional callback invoked before resetting a pending run.
                Receives the existing RunInfo. If it returns False, a new run is
                created instead. Defaults to always True (reset without prompting).

        Returns:
            RunInfo with the run's details (existing reset or newly created).

        Raises:
            ValidationError: If client was created with a glob-pattern context_key.
            NotFoundError: If based_on run doesn't exist.
        """
        if _is_pattern(self.context_key):
            raise ValidationError(
                "Cannot create runs with a glob-pattern context_key; use an exact context."
            )

        with self._session_factory() as session:
            # Check for existing pending run to reuse (only when not creating child run)
            if replace_stale and based_on is None:
                existing = self._find_pending_run(session)
                if existing is not None:
                    # Check callback - if False, fall through to create new
                    if on_replace is None or on_replace(RunInfo.from_model(existing)):
                        return self._reset_pending_run(session, existing)

            return self._create_new_run(session, adapter_name, config, based_on)

    def _create_new_run(
        self,
        session,
        adapter_name: str | None,
        config: dict | None,
        based_on: int | None,
    ) -> RunInfo:
        """Create a new training run in the database."""
        if based_on is not None:
            parent = self._get_run(session, based_on)
            if parent is None:
                raise NotFoundError(f"Parent run {based_on} not found")

        adapter = {"name": adapter_name} if adapter_name else None
        run = Run(
            method=self.method,
            context_key=self.context_key,
            adapter=adapter,
            based_on=based_on,
            config=config,
        )
        session.add(run)
        session.flush()
        info = RunInfo.from_model(run)
        self._lg.debug(
            "created training run",
            extra={"run_id": info.id, "method": self.method, "adapter_name": adapter_name},
        )
        return info

    def _find_pending_run(self, session) -> Run | None:
        """Find existing pending run for this context."""
        stmt = (
            select(Run)
            .where(Run.method == self.method)
            .where(Run.context_key == self.context_key)
            .where(Run.status == "pending")
            .where(_not_deleted_filter(Run))
            .order_by(Run.created_at.desc())
            .limit(1)
        )
        return cast(Run | None, session.scalar(stmt))

    def _reset_pending_run(self, session, run: Run) -> RunInfo:
        """Clear pairs from pending run and return it for reuse."""
        self._clear_pending_pairs(session, run.id)
        self._lg.debug(
            "reset pending run for reuse",
            extra={"run_id": run.id, "context_key": run.context_key},
        )
        return RunInfo.from_model(run)

    def get(self, run_id: int) -> RunInfo | None:
        """Get a training run by ID."""
        with self._session_factory() as session:
            run = self._get_run(session, run_id)
            if run is None:
                return None
            return RunInfo.from_model(run)

    def list_runs(
        self,
        status: str | None = None,
        limit: int = 100,
        descending: bool = True,
        include_deleted: bool = False,
    ) -> list[RunInfo]:
        """List DPO training runs.

        Args:
            status: Filter by status (pending, running, completed, failed).
            limit: Maximum number of runs to return.
            descending: Order by created_at descending (newest first).
            include_deleted: Include soft-deleted runs (default False).

        Returns:
            List of RunInfo.
        """
        if status is not None and status not in VALID_STATUSES:
            raise ValidationError(f"status must be one of {VALID_STATUSES}")

        with self._session_factory() as session:
            stmt = select(Run).where(Run.method == self.method)
            context_filter = self._build_context_filter(Run.context_key)
            if context_filter is not None:
                stmt = stmt.where(context_filter)
            if status is not None:
                stmt = stmt.where(Run.status == status)
            if not include_deleted:
                stmt = stmt.where(_not_deleted_filter(Run))
            order = Run.created_at.desc() if descending else Run.created_at.asc()
            stmt = stmt.order_by(order).limit(limit)

            runs = list(session.scalars(stmt).all())
            return [RunInfo.from_model(r) for r in runs]

    def delete(self, run_id: int) -> bool:
        """Soft-delete a training run (frees pending pairs for reuse)."""
        with self._session_factory() as session:
            run = self._get_run(session, run_id)
            if run is None:
                return False

            # Clear pending pairs (they become available for other runs)
            self._clear_pending_pairs(session, run_id)

            # Soft-delete via system_status
            run.system_status = {"deleted": True, "deleted_at": utc_now().isoformat()}
            self._lg.debug("soft-deleted training run", extra={"run_id": run_id})
            return True

    # -------------------------------------------------------------------------
    # Pair assignment
    # -------------------------------------------------------------------------

    def assign_pairs(self, run_id: int, pairs: Sequence[PairTuple]) -> int:
        """Assign preference pairs to a DPO run.

        Pairs that have been used in this run's lineage are automatically filtered out.

        Args:
            run_id: The run ID to assign pairs to.
            pairs: List of (chosen_fact_id, rejected_fact_id, prompt) tuples.

        Returns:
            Number of pairs assigned (may be less than input if some filtered by lineage).

        Raises:
            NotFoundError: If run_id doesn't exist or isn't accessible.
        """
        if not pairs:
            return 0

        with self._session_factory() as session:
            run = self._get_run(session, run_id)
            if run is None:
                raise NotFoundError(f"Training run {run_id} not found")

            # Filter out pairs already used in lineage
            lineage_pairs = self._get_lineage_pairs(session, run_id)
            new_pairs = [p for p in pairs if (p[0], p[1]) not in lineage_pairs]

            if not new_pairs:
                self._lg.debug(
                    "all pairs filtered by lineage",
                    extra={"run_id": run_id, "input_count": len(pairs)},
                )
                return 0

            self._insert_pending_pairs(session, run_id, new_pairs)

            self._lg.debug(
                "assigned pairs to run",
                extra={
                    "run_id": run_id,
                    "assigned": len(new_pairs),
                    "filtered": len(pairs) - len(new_pairs),
                },
            )
            return len(new_pairs)

    def _get_lineage_pairs(self, session, run_id: int) -> set[tuple[int, int]]:
        """Get all pairs used in ancestor runs (for lineage-based exclusion)."""
        # Walk the lineage chain using recursive CTE

        # Build recursive CTE manually
        anchor = select(Run.id, Run.based_on).where(Run.id == run_id)
        lineage_cte = anchor.cte(name="lineage", recursive=True)

        recursive = select(Run.id, Run.based_on).where(Run.id == lineage_cte.c.based_on)
        lineage_cte = lineage_cte.union_all(recursive)

        # Get trained pairs from all runs in lineage
        stmt = select(TrainedPair.chosen_fact_id, TrainedPair.rejected_fact_id).where(
            TrainedPair.run_id.in_(select(lineage_cte.c.id))
        )

        results = session.execute(stmt).all()
        return {(row[0], row[1]) for row in results}

    def _insert_pending_pairs(self, session, run_id: int, pairs: Sequence[PairTuple]) -> None:
        """Insert pairs into pending table."""
        for chosen_id, rejected_id, prompt in pairs:
            session.add(
                PendingPair(
                    run_id=run_id,
                    chosen_fact_id=chosen_id,
                    rejected_fact_id=rejected_id,
                    prompt=prompt,
                )
            )
        session.flush()

    def _clear_pending_pairs(self, session, run_id: int) -> int:
        """Delete all pending pairs for a run."""
        from sqlalchemy import delete

        stmt = delete(PendingPair).where(PendingPair.run_id == run_id)
        result = session.execute(stmt)
        return int(result.rowcount)

    def _move_pairs_to_trained(self, session, run_id: int) -> int:
        """Move pairs from pending to trained (on completion)."""
        # Get pending pairs
        stmt = select(PendingPair).where(PendingPair.run_id == run_id)
        pending = list(session.scalars(stmt).all())

        if not pending:
            return 0

        # Insert into trained
        now = utc_now()
        for pair in pending:
            session.add(
                TrainedPair(
                    run_id=run_id,
                    chosen_fact_id=pair.chosen_fact_id,
                    rejected_fact_id=pair.rejected_fact_id,
                    prompt=pair.prompt,
                    trained_at=now,
                )
            )

        # Delete from pending
        from sqlalchemy import delete

        session.execute(delete(PendingPair).where(PendingPair.run_id == run_id))

        return len(pending)

    def get_pairs(self, run_id: int) -> list[PairTuple]:
        """Get all pending pairs assigned to a run.

        Args:
            run_id: The run ID.

        Returns:
            List of (chosen_fact_id, rejected_fact_id, prompt) tuples.

        Raises:
            NotFoundError: If run_id doesn't exist or isn't accessible.
        """
        with self._session_factory() as session:
            run = self._get_run(session, run_id)
            if run is None:
                raise NotFoundError(f"Training run {run_id} not found")

            stmt = (
                select(
                    PendingPair.chosen_fact_id,
                    PendingPair.rejected_fact_id,
                    PendingPair.prompt,
                )
                .where(PendingPair.run_id == run_id)
                .order_by(PendingPair.assigned_at)
            )

            results = session.execute(stmt).all()
            return [(r[0], r[1], r[2]) for r in results]

    def get_trained_pairs(self, run_id: int) -> list[PairTuple]:
        """Get all trained pairs for a completed run.

        Args:
            run_id: The run ID.

        Returns:
            List of (chosen_fact_id, rejected_fact_id, prompt) tuples.

        Raises:
            NotFoundError: If run_id doesn't exist or isn't accessible.
        """
        with self._session_factory() as session:
            run = self._get_run(session, run_id)
            if run is None:
                raise NotFoundError(f"Training run {run_id} not found")

            stmt = (
                select(
                    TrainedPair.chosen_fact_id,
                    TrainedPair.rejected_fact_id,
                    TrainedPair.prompt,
                )
                .where(TrainedPair.run_id == run_id)
                .order_by(TrainedPair.trained_at)
            )

            results = session.execute(stmt).all()
            return [(r[0], r[1], r[2]) for r in results]

    def count_pending_pairs(self, run_id: int | None = None) -> int:
        """Count pending pairs for a run or all runs in context."""
        with self._session_factory() as session:
            from sqlalchemy import func

            stmt = select(func.count()).select_from(PendingPair)

            if run_id is not None:
                stmt = stmt.where(PendingPair.run_id == run_id)
            else:
                # Count for all runs in context
                subq = select(Run.id).where(
                    Run.method == self.method,
                    _not_deleted_filter(Run),
                )
                context_filter = self._build_context_filter(Run.context_key)
                if context_filter is not None:
                    subq = subq.where(context_filter)
                stmt = stmt.where(PendingPair.run_id.in_(subq))

            return session.scalar(stmt) or 0

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    def start(self, run_id: int) -> None:
        """Mark a training run as started."""
        self._transition(run_id, "running")

    def complete(self, run_id: int, metrics: dict | None = None) -> None:
        """Mark a training run as completed and move pairs to trained."""
        with self._session_factory() as session:
            run = self._get_run(session, run_id)
            if run is None:
                raise NotFoundError(f"Training run {run_id} not found")

            allowed = STATUS_TRANSITIONS.get(run.status, set())
            if "completed" not in allowed:
                raise ValidationError(
                    f"Cannot transition from '{run.status}' to 'completed'. "
                    f"Allowed transitions: {allowed or 'none (terminal state)'}"
                )

            # Move pairs to trained table
            pairs_moved = self._move_pairs_to_trained(session, run_id)

            # Update run
            run.status = "completed"
            run.completed_at = utc_now()
            if metrics is not None:
                run.metrics = metrics

            self._lg.debug(
                "completed training run",
                extra={"run_id": run_id, "pairs_trained": pairs_moved},
            )

    def fail(self, run_id: int, error: str) -> None:
        """Mark a training run as failed (clears pending pairs for reuse)."""
        with self._session_factory() as session:
            run = self._get_run(session, run_id)
            if run is None:
                raise NotFoundError(f"Training run {run_id} not found")

            allowed = STATUS_TRANSITIONS.get(run.status, set())
            if "failed" not in allowed:
                raise ValidationError(
                    f"Cannot transition from '{run.status}' to 'failed'. "
                    f"Allowed transitions: {allowed or 'none (terminal state)'}"
                )

            # Clear pending pairs (they become available for other runs)
            self._clear_pending_pairs(session, run_id)

            run.status = "failed"
            run.completed_at = utc_now()
            run.error_message = error

            self._lg.debug("failed training run", extra={"run_id": run_id, "error": error})

    def reset(self, run_id: int) -> None:
        """Reset a running training run back to pending (for retry after transient failure)."""
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
                raise NotFoundError(f"Training run {run_id} not found")

            allowed = STATUS_TRANSITIONS.get(run.status, set())
            if new_status not in allowed:
                raise ValidationError(
                    f"Cannot transition from '{run.status}' to '{new_status}'. "
                    f"Allowed transitions: {allowed or 'none (terminal state)'}"
                )

            self._apply_transition(run, new_status, metrics, error_message, clear_started)
            self._lg.debug(
                "training run status changed", extra={"run_id": run_id, "new_status": new_status}
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
        """Soft-delete all training runs for this context (clears pending pairs)."""
        with self._session_factory() as session:
            stmt = select(Run).where(
                Run.method == self.method,
                _not_deleted_filter(Run),
            )
            context_filter = self._build_context_filter(Run.context_key)
            if context_filter is not None:
                stmt = stmt.where(context_filter)

            runs = list(session.scalars(stmt).all())
            count = len(runs)

            now = utc_now().isoformat()
            for run in runs:
                self._clear_pending_pairs(session, run.id)
                run.system_status = {"deleted": True, "deleted_at": now}

            self._lg.info("reset all training runs", extra={"count": count})
            return count

    def clear_pending_pairs_for_context(self) -> int:
        """Clear all pending pairs for this context (without deleting runs)."""
        with self._session_factory() as session:
            from sqlalchemy import delete

            # Get run IDs for this context
            run_stmt = select(Run.id).where(
                Run.method == self.method,
                _not_deleted_filter(Run),
            )
            context_filter = self._build_context_filter(Run.context_key)
            if context_filter is not None:
                run_stmt = run_stmt.where(context_filter)

            run_ids = list(session.scalars(run_stmt).all())
            if not run_ids:
                return 0

            stmt = delete(PendingPair).where(PendingPair.run_id.in_(run_ids))
            result = session.execute(stmt)
            count = int(result.rowcount)

            self._lg.info("cleared pending pairs", extra={"count": count})
            return count


def _is_pattern(context_key: str | None) -> bool:
    """Check if context_key contains glob pattern characters."""
    if context_key is None:
        return False
    return "*" in context_key or "?" in context_key
