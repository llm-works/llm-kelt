"""Rename generic table names to avoid conflicts with agent tables.

Renames:
- sessions -> conv_sessions
- embeddings -> fact_embeddings

Revision ID: 004
Revises: 003
Create Date: 2026-04-19

"""

from collections.abc import Sequence

from alembic import op

revision: str = "004"
down_revision: str = "003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:  # cq: exempt
    # =========================================================================
    # Rename sessions -> conv_sessions
    # =========================================================================

    # Rename table
    op.rename_table("sessions", "conv_sessions")

    # Rename indexes in place (no rebuild)
    op.execute("ALTER INDEX idx_sessions_session_id RENAME TO idx_conv_sessions_session_id")
    op.execute("ALTER INDEX idx_sessions_updated_at RENAME TO idx_conv_sessions_updated_at")

    # Rename constraint in place
    op.execute(
        "ALTER TABLE conv_sessions RENAME CONSTRAINT "
        "uq_sessions_session_id TO uq_conv_sessions_session_id"
    )

    # =========================================================================
    # Rename embeddings -> fact_embeddings
    # =========================================================================

    # Rename table
    op.rename_table("embeddings", "fact_embeddings")

    # Rename indexes in place (no rebuild, including HNSW vector index)
    op.execute("ALTER INDEX idx_embedding_entity RENAME TO idx_fact_embedding_entity")
    op.execute("ALTER INDEX idx_embedding_model RENAME TO idx_fact_embedding_model")
    op.execute("ALTER INDEX idx_embeddings_vector_1536 RENAME TO idx_fact_embeddings_vector_1536")

    # Rename constraint in place
    op.execute(
        "ALTER TABLE fact_embeddings RENAME CONSTRAINT "
        "uq_embedding_entity_model TO uq_fact_embedding_entity_model"
    )


def downgrade() -> None:  # cq: exempt
    # =========================================================================
    # Rename fact_embeddings -> embeddings
    # =========================================================================

    # Rename constraint back
    op.execute(
        "ALTER TABLE fact_embeddings RENAME CONSTRAINT "
        "uq_fact_embedding_entity_model TO uq_embedding_entity_model"
    )

    # Rename indexes back
    op.execute("ALTER INDEX idx_fact_embedding_entity RENAME TO idx_embedding_entity")
    op.execute("ALTER INDEX idx_fact_embedding_model RENAME TO idx_embedding_model")
    op.execute("ALTER INDEX idx_fact_embeddings_vector_1536 RENAME TO idx_embeddings_vector_1536")

    # Rename table
    op.rename_table("fact_embeddings", "embeddings")

    # =========================================================================
    # Rename conv_sessions -> sessions
    # =========================================================================

    # Rename constraint back
    op.execute(
        "ALTER TABLE conv_sessions RENAME CONSTRAINT "
        "uq_conv_sessions_session_id TO uq_sessions_session_id"
    )

    # Rename indexes back
    op.execute("ALTER INDEX idx_conv_sessions_session_id RENAME TO idx_sessions_session_id")
    op.execute("ALTER INDEX idx_conv_sessions_updated_at RENAME TO idx_sessions_updated_at")

    # Rename table
    op.rename_table("conv_sessions", "sessions")
