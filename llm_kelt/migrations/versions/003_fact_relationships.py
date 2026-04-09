"""Add fact relationships table for graph-like edges between atomic facts.

Revision ID: 003
Revises: 002
Create Date: 2026-04-09

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "003"
down_revision: str = "002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:  # cq: exempt
    op.create_table(
        "atomic_fact_relationships",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("source_id", sa.BigInteger(), nullable=False),
        sa.Column("target_id", sa.BigInteger(), nullable=False),
        sa.Column("relationship_type", sa.String(50), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=True, server_default="1.0"),
        sa.Column("metadata", postgresql.JSONB(), nullable=True),
        sa.Column("context_key", sa.String(255), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["source_id"], ["atomic_facts.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["target_id"], ["atomic_facts.id"], ondelete="CASCADE"),
        sa.UniqueConstraint(
            "source_id", "target_id", "relationship_type", name="uq_atomic_rel_src_tgt_type"
        ),
    )
    op.create_index(
        "idx_atomic_rel_target_type",
        "atomic_fact_relationships",
        ["target_id", "relationship_type"],
    )
    op.create_index(
        "idx_atomic_rel_context",
        "atomic_fact_relationships",
        ["context_key"],
    )
    op.create_index(
        "idx_atomic_rel_type",
        "atomic_fact_relationships",
        ["relationship_type"],
    )


def downgrade() -> None:
    op.drop_index("idx_atomic_rel_type", table_name="atomic_fact_relationships")
    op.drop_index("idx_atomic_rel_context", table_name="atomic_fact_relationships")
    op.drop_index("idx_atomic_rel_target_type", table_name="atomic_fact_relationships")
    op.drop_table("atomic_fact_relationships")
