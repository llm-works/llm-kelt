"""SFT training client with example management.

Manages SFT training runs:
- Creating training runs (with lineage via based_on)
- Assigning feedback examples to runs (lineage-aware - examples excluded from ancestor runs)
- Tracking run lifecycle (pending → running → completed/failed)
- Moving examples from pending to trained on completion

SFT is simpler than DPO: trains on individual positive feedback examples, not preference pairs.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import datetime
from typing import Any, cast

from appinfra.log import Logger
from sqlalchemy import BigInteger, DateTime, ForeignKey, Index, Table, select
from sqlalchemy.orm import Mapped, mapped_column

from llm_learn.core.base import Base, utc_now
from llm_learn.core.exceptions import NotFoundError, ValidationError
from llm_learn.memory.isolation import build_context_filter

from ..models import (
    STATUS_TRANSITIONS,
    VALID_STATUSES,
    Run,
    RunInfo,
    _is_pattern,
    _not_deleted_filter,
    get_parent_run,
)

# =============================================================================
# ORM Models for SFT
# =============================================================================


class PendingExample(Base):
    """Temporary table for examples assigned to pending/running SFT runs.

    Examples reference feedback facts via fact_id.
    Deleted when run completes (moved to trained) or fails (freed for retry).
    """

    __tablename__ = "sft_pending_examples"

    run_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("training_runs.id", ondelete="CASCADE"), primary_key=True
    )
    fact_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("atomic_facts.id", ondelete="CASCADE"), primary_key=True
    )
    assigned_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )

    __table_args__ = (Index("idx_sft_pending_examples_run", "run_id"),)

    def __repr__(self) -> str:
        return f"<PendingExample(run={self.run_id}, fact={self.fact_id})>"


class TrainedExample(Base):
    """Permanent history of examples used in completed SFT training.

    Examples are moved here when a run completes successfully.
    Used for lineage-based exclusion in future runs.
    """

    __tablename__ = "sft_trained_examples"

    run_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("training_runs.id", ondelete="CASCADE"), primary_key=True
    )
    fact_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("atomic_facts.id", ondelete="CASCADE"), primary_key=True
    )
    trained_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )

    __table_args__ = (Index("idx_sft_trained_examples_run", "run_id"),)

    def __repr__(self) -> str:
        return f"<TrainedExample(run={self.run_id}, fact={self.fact_id})>"


# =============================================================================
# Client
# =============================================================================


class Client:
    """Client for SFT training with example management.

    Manages:
    - Creating/listing/deleting SFT training runs
    - Assigning feedback examples to runs (lineage-aware)
    - Tracking run lifecycle (pending → running → completed/failed)
    - Moving examples from pending to trained on completion
    """

    def __init__(
        self,
        lg: Logger,
        session_factory: Callable[[], Any],
        context_key: str | None,
        ensure_schema: bool = False,
    ) -> None:
        """Initialize SFT client.

        Args:
            lg: Logger instance.
            session_factory: Callable that returns a context manager for database sessions.
            context_key: Context key to scope operations (None = no filtering).
            ensure_schema: If True, create training tables if they don't exist.
        """
        self._lg = lg
        self._session_factory = session_factory
        self.context_key = context_key
        self.method = "sft"

        if ensure_schema:
            self._ensure_tables()

    def _ensure_tables(self) -> None:
        """Create training tables if they don't exist."""
        with self._session_factory() as session:
            engine = session.get_bind()
            tables = [
                cast(Table, Run.__table__),
                cast(Table, PendingExample.__table__),
                cast(Table, TrainedExample.__table__),
            ]
            Base.metadata.create_all(engine, tables=tables)
            self._lg.debug("ensured SFT training tables exist")

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
        """Create a new SFT training run, or reset an existing pending run.

        If a pending run already exists for this context and `replace_stale=True`,
        clears its examples and returns it (no new run created). This allows callers
        to safely retry run creation without accumulating orphan runs.

        Args:
            adapter_name: Optional name for the adapter being trained.
            config: Optional training configuration dict.
            based_on: Optional run ID to base this run on (for lineage).
            replace_stale: If True (default), reuse any existing pending run for
                this context by clearing its examples. If False, always create new.
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
                        return self._reset_pending_run(session, existing, adapter_name, config)

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
            parent = get_parent_run(session, based_on, self.context_key)
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
            "created SFT training run",
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

    def _reset_pending_run(
        self,
        session,
        run: Run,
        adapter_name: str | None = None,
        config: dict | None = None,
    ) -> RunInfo:
        """Clear examples from pending run, update parameters, and return for reuse."""
        self._clear_pending_examples(session, run.id)

        # Update run with caller's parameters
        if adapter_name is not None:
            run.adapter = {"name": adapter_name}
        if config is not None:
            run.config = config
        session.flush()

        self._lg.debug(
            "reset pending SFT run for reuse",
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
        """List SFT training runs.

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
        """Delete a training run.

        For pending runs: transitions to 'cancelled' status (proper state change).
        For other runs: soft-deletes via system_status (hides without changing history).

        In both cases, pending examples are cleared and become available for reuse.
        """
        with self._session_factory() as session:
            run = self._get_run(session, run_id)
            if run is None:
                return False

            # Clear pending examples (they become available for other runs)
            self._clear_pending_examples(session, run_id)

            if run.status == "pending":
                # Pending runs can be properly cancelled (remain visible with cancelled status)
                run.status = "cancelled"
                self._lg.debug("cancelled pending SFT run", extra={"run_id": run_id})
            else:
                # Running/completed/failed runs are soft-deleted (preserve history)
                run.system_status = {"deleted": True, "deleted_at": utc_now().isoformat()}
                self._lg.debug("soft-deleted SFT training run", extra={"run_id": run_id})
            return True

    # -------------------------------------------------------------------------
    # Example assignment
    # -------------------------------------------------------------------------

    def _validate_pending_status(self, run: Run) -> None:
        """Validate run is in pending status for modification."""
        if run.status != "pending":
            raise ValidationError(f"Cannot modify run in '{run.status}' status (must be 'pending')")

    def _filter_and_assign_examples(self, session, run_id: int, fact_ids: Sequence[int]) -> int:
        """Filter examples by lineage and insert. Returns count assigned."""
        lineage_examples = self._get_lineage_examples(session, run_id)
        new_fact_ids = [fid for fid in fact_ids if fid not in lineage_examples]

        if not new_fact_ids:
            self._lg.debug(
                "all examples filtered by lineage",
                extra={"run_id": run_id, "input_count": len(fact_ids)},
            )
            return 0

        self._insert_pending_examples(session, run_id, new_fact_ids)
        self._lg.debug(
            "assigned examples to SFT run",
            extra={"run_id": run_id, "assigned": len(new_fact_ids)},
        )
        return len(new_fact_ids)

    def assign_examples(self, run_id: int, fact_ids: Sequence[int]) -> int:
        """Assign feedback examples to an SFT run.

        Examples that have been used in this run's lineage are automatically filtered out.

        Args:
            run_id: The run ID to assign examples to.
            fact_ids: List of fact IDs (from atomic_facts) to assign.

        Returns:
            Number of examples assigned (may be less than input if some filtered by lineage).

        Raises:
            NotFoundError: If run_id doesn't exist or isn't accessible.
            ValidationError: If run is not in 'pending' status.
        """
        if not fact_ids:
            return 0

        with self._session_factory() as session:
            run = self._get_run(session, run_id)
            if run is None:
                raise NotFoundError(f"Training run {run_id} not found")
            self._validate_pending_status(run)
            return self._filter_and_assign_examples(session, run_id, fact_ids)

    def _get_lineage_examples(self, session, run_id: int) -> set[int]:
        """Get all examples used in ancestor runs (for lineage-based exclusion)."""
        # Walk the lineage chain using recursive CTE
        anchor = select(Run.id, Run.based_on).where(Run.id == run_id)
        lineage_cte = anchor.cte(name="lineage", recursive=True)

        recursive = select(Run.id, Run.based_on).where(Run.id == lineage_cte.c.based_on)
        lineage_cte = lineage_cte.union_all(recursive)

        # Get trained examples from all runs in lineage
        stmt = select(TrainedExample.fact_id).where(
            TrainedExample.run_id.in_(select(lineage_cte.c.id))
        )

        results = session.execute(stmt).all()
        return {row[0] for row in results}

    def _insert_pending_examples(self, session, run_id: int, fact_ids: Sequence[int]) -> None:
        """Insert examples into pending table."""
        for fact_id in fact_ids:
            session.add(PendingExample(run_id=run_id, fact_id=fact_id))
        session.flush()

    def _clear_pending_examples(self, session, run_id: int) -> int:
        """Delete all pending examples for a run."""
        from sqlalchemy import delete

        stmt = delete(PendingExample).where(PendingExample.run_id == run_id)
        result = session.execute(stmt)
        return int(result.rowcount)

    def _move_examples_to_trained(self, session, run_id: int) -> int:
        """Move examples from pending to trained (on completion)."""
        # Get pending examples
        stmt = select(PendingExample).where(PendingExample.run_id == run_id)
        pending = list(session.scalars(stmt).all())

        if not pending:
            return 0

        # Insert into trained
        now = utc_now()
        for example in pending:
            session.add(
                TrainedExample(
                    run_id=run_id,
                    fact_id=example.fact_id,
                    trained_at=now,
                )
            )

        # Delete from pending
        from sqlalchemy import delete

        session.execute(delete(PendingExample).where(PendingExample.run_id == run_id))

        return len(pending)

    def get_examples(self, run_id: int) -> list[int]:
        """Get all pending examples assigned to a run.

        Args:
            run_id: The run ID.

        Returns:
            List of fact IDs.

        Raises:
            NotFoundError: If run_id doesn't exist or isn't accessible.
        """
        with self._session_factory() as session:
            run = self._get_run(session, run_id)
            if run is None:
                raise NotFoundError(f"Training run {run_id} not found")

            stmt = (
                select(PendingExample.fact_id)
                .where(PendingExample.run_id == run_id)
                .order_by(PendingExample.assigned_at)
            )

            results = session.execute(stmt).all()
            return [r[0] for r in results]

    def get_trained_examples(self, run_id: int) -> list[int]:
        """Get all trained examples for a completed run.

        Args:
            run_id: The run ID.

        Returns:
            List of fact IDs.

        Raises:
            NotFoundError: If run_id doesn't exist or isn't accessible.
        """
        with self._session_factory() as session:
            run = self._get_run(session, run_id)
            if run is None:
                raise NotFoundError(f"Training run {run_id} not found")

            stmt = (
                select(TrainedExample.fact_id)
                .where(TrainedExample.run_id == run_id)
                .order_by(TrainedExample.trained_at)
            )

            results = session.execute(stmt).all()
            return [r[0] for r in results]

    def count_pending_examples(self, run_id: int | None = None) -> int:
        """Count pending examples for a run or all runs in context."""
        with self._session_factory() as session:
            from sqlalchemy import func

            stmt = select(func.count()).select_from(PendingExample)

            if run_id is not None:
                stmt = stmt.where(PendingExample.run_id == run_id)
            else:
                # Count for all runs in context
                subq = select(Run.id).where(
                    Run.method == self.method,
                    _not_deleted_filter(Run),
                )
                context_filter = self._build_context_filter(Run.context_key)
                if context_filter is not None:
                    subq = subq.where(context_filter)
                stmt = stmt.where(PendingExample.run_id.in_(subq))

            return session.scalar(stmt) or 0

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    def start(self, run_id: int) -> None:
        """Mark a training run as started."""
        self._transition(run_id, "running")

    def complete(
        self,
        run_id: int,
        metrics: dict | None = None,
        adapter_info: dict | None = None,
    ) -> None:
        """Mark a training run as completed and move examples to trained.

        Args:
            run_id: The run to complete.
            metrics: Optional training metrics to store.
            adapter_info: Optional adapter metadata (mtime, md5) to merge into adapter field.
        """
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

            # Move examples to trained table
            examples_moved = self._move_examples_to_trained(session, run_id)

            # Update run
            run.status = "completed"
            run.completed_at = utc_now()
            if metrics is not None:
                run.metrics = metrics
            if adapter_info is not None:
                # Merge adapter_info into existing adapter field
                run.adapter = {**(run.adapter or {}), **adapter_info}

            self._lg.debug(
                "completed SFT training run",
                extra={"run_id": run_id, "examples_trained": examples_moved},
            )

    def fail(self, run_id: int, error: str) -> None:
        """Mark a training run as failed (clears pending examples for reuse)."""
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

            # Clear pending examples (they become available for other runs)
            self._clear_pending_examples(session, run_id)

            run.status = "failed"
            run.completed_at = utc_now()
            run.error_message = error

            self._lg.debug("failed SFT training run", extra={"run_id": run_id, "error": error})

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
                "SFT training run status changed",
                extra={"run_id": run_id, "new_status": new_status},
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
        """Soft-delete all training runs for this context (clears pending examples)."""
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
                self._clear_pending_examples(session, run.id)
                run.system_status = {"deleted": True, "deleted_at": now}

            self._lg.info("reset all SFT training runs", extra={"count": count})
            return count

    def clear_pending_examples_for_context(self) -> int:
        """Clear all pending examples for this context (without deleting runs)."""
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

            stmt = delete(PendingExample).where(PendingExample.run_id.in_(run_ids))
            result = session.execute(stmt)
            count = int(result.rowcount)

            self._lg.info("cleared pending SFT examples", extra={"count": count})
            return count
