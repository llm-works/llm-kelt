# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-kelt Authors

"""Drop legacy fact_embeddings table in favor of quantized format-specific tables.

Revision ID: 006
Revises: 005
Create Date: 2025-05-17

New embedding tables are created on-demand by the application:
- embeddings_{dim}_f32 (pgvector vector)
- embeddings_{dim}_f16 (pgvector halfvec)
- embeddings_{dim}_i8 (bytea + scalar quantization)
- embeddings_{dim}_i4 (bytea + packed 4-bit quantization)
"""

from collections.abc import Sequence

from alembic import op

revision: str = "006"
down_revision: str = "005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Use IF EXISTS - indexes/table may not exist on all deployments
    op.execute("DROP INDEX IF EXISTS idx_fact_embeddings_vector_1536")
    op.execute("DROP INDEX IF EXISTS idx_fact_embedding_model")
    op.execute("DROP INDEX IF EXISTS idx_fact_embedding_entity")
    op.execute("DROP TABLE IF EXISTS fact_embeddings")


def downgrade() -> None:
    raise NotImplementedError(
        "Downgrade not supported - embedding data would be lost. Restore from backup if needed."
    )
