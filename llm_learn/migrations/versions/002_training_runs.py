"""Add training run tracking tables.

Revision ID: 002
Revises: 001
Create Date: 2025-02-14

This migration adds tables for tracking DPO training runs and
which preference pairs have been used in each run.
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
    """Add training run tracking tables."""
    op.create_table(
        "training_runs",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("context_key", sa.String(255), nullable=True),
        sa.Column("adapter_name", sa.String(255), nullable=True),
        sa.Column("status", sa.String(20), server_default="pending", nullable=False),
        sa.Column("config", postgresql.JSONB(), nullable=True),
        sa.Column("metrics", postgresql.JSONB(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_training_runs_context", "training_runs", ["context_key"])
    op.create_index("idx_training_runs_status", "training_runs", ["status"])
    op.create_index("idx_training_runs_adapter", "training_runs", ["adapter_name"])
    op.create_index("idx_training_runs_created", "training_runs", ["created_at"])
    # Prefix index for efficient LIKE queries (pattern matching)
    op.create_index(
        "idx_training_runs_context_prefix",
        "training_runs",
        ["context_key"],
        postgresql_ops={"context_key": "varchar_pattern_ops"},
    )

    op.create_table(
        "training_run_pairs",
        sa.Column("training_run_id", sa.BigInteger(), nullable=False),
        sa.Column("preference_fact_id", sa.BigInteger(), nullable=False),
        sa.Column(
            "assigned_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.ForeignKeyConstraint(["training_run_id"], ["training_runs.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["preference_fact_id"], ["atomic_facts.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("training_run_id", "preference_fact_id"),
        # UNIQUE constraint enforces exclusive assignment - each pair can only be in ONE run
        sa.UniqueConstraint("preference_fact_id", name="uq_training_run_pair_exclusive"),
    )
    op.create_index("idx_training_run_pairs_run", "training_run_pairs", ["training_run_id"])
    op.create_index("idx_training_run_pairs_fact", "training_run_pairs", ["preference_fact_id"])


def downgrade() -> None:
    """Remove training run tracking tables."""
    op.drop_table("training_run_pairs")
    op.drop_table("training_runs")
