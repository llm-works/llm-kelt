"""Add training tables.

Revision ID: 002
Revises: 001
Create Date: 2025-02-14

This migration adds tables for tracking training runs and pairs:
- training_runs: Generic training run metadata (supports DPO, SFT, RLHF)
- dpo_pending_pairs: Pairs assigned to pending/running DPO runs
- dpo_trained_pairs: Pairs used in completed DPO training (permanent history)
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "002"
down_revision: str = "001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:  # cq: exempt
    """Add training tables."""
    # training_runs - generic training run metadata
    op.create_table(
        "training_runs",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("method", sa.String(20), nullable=False),
        sa.Column("context_key", sa.String(255), nullable=True),
        sa.Column("adapter", postgresql.JSONB(), nullable=True),
        sa.Column("based_on", sa.BigInteger(), nullable=True),
        sa.Column("status", sa.String(20), server_default="pending", nullable=False),
        sa.Column("config", postgresql.JSONB(), nullable=True),
        sa.Column("metrics", postgresql.JSONB(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("system_status", postgresql.JSONB(), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["based_on"], ["training_runs.id"], ondelete="SET NULL"),
    )
    op.create_index("idx_training_runs_method", "training_runs", ["method"])
    op.create_index("idx_training_runs_context", "training_runs", ["context_key"])
    op.create_index("idx_training_runs_status", "training_runs", ["status"])
    op.create_index("idx_training_runs_created", "training_runs", ["created_at"])
    op.create_index("idx_training_runs_based_on", "training_runs", ["based_on"])
    op.create_index(
        "idx_training_runs_context_prefix",
        "training_runs",
        ["context_key"],
        postgresql_ops={"context_key": "varchar_pattern_ops"},
    )
    # Expression index for adapter name lookups
    op.create_index(
        "idx_training_runs_adapter_name",
        "training_runs",
        [sa.text("(adapter->>'name')")],
        postgresql_using="btree",
    )

    # dpo_pending_pairs - temporary pairs for pending/running runs
    op.create_table(
        "dpo_pending_pairs",
        sa.Column("run_id", sa.BigInteger(), nullable=False),
        sa.Column("chosen_fact_id", sa.BigInteger(), nullable=False),
        sa.Column("rejected_fact_id", sa.BigInteger(), nullable=False),
        sa.Column("prompt", sa.Text(), nullable=False),
        sa.Column(
            "assigned_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.ForeignKeyConstraint(["run_id"], ["training_runs.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["chosen_fact_id"], ["atomic_facts.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["rejected_fact_id"], ["atomic_facts.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("run_id", "chosen_fact_id", "rejected_fact_id"),
    )
    op.create_index("idx_dpo_pending_pairs_run", "dpo_pending_pairs", ["run_id"])

    # dpo_trained_pairs - permanent history of trained pairs
    op.create_table(
        "dpo_trained_pairs",
        sa.Column("run_id", sa.BigInteger(), nullable=False),
        sa.Column("chosen_fact_id", sa.BigInteger(), nullable=False),
        sa.Column("rejected_fact_id", sa.BigInteger(), nullable=False),
        sa.Column("prompt", sa.Text(), nullable=False),
        sa.Column(
            "trained_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.ForeignKeyConstraint(["run_id"], ["training_runs.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["chosen_fact_id"], ["atomic_facts.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["rejected_fact_id"], ["atomic_facts.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("run_id", "chosen_fact_id", "rejected_fact_id"),
    )
    op.create_index("idx_dpo_trained_pairs_run", "dpo_trained_pairs", ["run_id"])


def downgrade() -> None:
    """Remove training tables."""
    op.drop_table("dpo_trained_pairs")
    op.drop_table("dpo_pending_pairs")
    op.drop_table("training_runs")
