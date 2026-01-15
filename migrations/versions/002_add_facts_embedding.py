"""Add fact_embeddings table for vector search.

Revision ID: 002
Revises: 001
Create Date: 2026-01-14

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

# revision identifiers, used by Alembic.
revision: str = "002"
down_revision: str | None = "001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Create fact_embeddings table (matches FactEmbedding model)
    op.create_table(
        "fact_embeddings",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("fact_id", sa.BigInteger(), nullable=False),
        sa.Column("model_name", sa.String(100), nullable=False),
        sa.Column("dimensions", sa.Integer(), nullable=False),
        sa.Column("embedding", Vector(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["fact_id"], ["facts.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("fact_id", "model_name", name="uq_fact_embedding_model"),
    )
    op.create_index("idx_fact_embeddings_fact", "fact_embeddings", ["fact_id"])
    op.create_index("idx_fact_embeddings_model", "fact_embeddings", ["model_name"])

    # Create HNSW index for vector similarity search (better for smaller datasets)
    op.execute(
        """
        CREATE INDEX idx_fact_embeddings_vector ON fact_embeddings
        USING hnsw (embedding vector_cosine_ops)
        """
    )


def downgrade() -> None:
    op.drop_index("idx_fact_embeddings_vector", table_name="fact_embeddings")
    op.drop_index("idx_fact_embeddings_model", table_name="fact_embeddings")
    op.drop_index("idx_fact_embeddings_fact", table_name="fact_embeddings")
    op.drop_table("fact_embeddings")
