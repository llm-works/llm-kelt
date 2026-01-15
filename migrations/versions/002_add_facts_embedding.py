"""Add embedding column to facts table for vector search.

Revision ID: 002
Revises: 001
Create Date: 2026-01-14

"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "002"
down_revision: str | None = "001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Add embedding column to facts table
    op.execute("ALTER TABLE facts ADD COLUMN embedding vector(1536)")

    # Create IVFFlat index for approximate nearest neighbor search
    op.execute(
        """
        CREATE INDEX idx_facts_embedding ON facts
        USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100)
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_facts_embedding")
    op.execute("ALTER TABLE facts DROP COLUMN IF EXISTS embedding")
