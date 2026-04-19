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

    # Drop old indexes
    op.drop_index("idx_sessions_session_id", table_name="sessions")
    op.drop_index("idx_sessions_updated_at", table_name="sessions")

    # Rename table
    op.rename_table("sessions", "conv_sessions")

    # Recreate indexes with new names
    op.create_index("idx_conv_sessions_session_id", "conv_sessions", ["session_id"])
    op.create_index("idx_conv_sessions_updated_at", "conv_sessions", ["updated_at"])

    # Rename the unique constraint (drop and recreate since Postgres doesn't support rename)
    op.drop_constraint("uq_sessions_session_id", "conv_sessions", type_="unique")
    op.create_unique_constraint("uq_conv_sessions_session_id", "conv_sessions", ["session_id"])

    # =========================================================================
    # Rename embeddings -> fact_embeddings
    # =========================================================================

    # Drop HNSW vector index first (references table by name)
    op.execute("DROP INDEX IF EXISTS idx_embeddings_vector_1536")

    # Drop old indexes
    op.drop_index("idx_embedding_entity", table_name="embeddings")
    op.drop_index("idx_embedding_model", table_name="embeddings")

    # Rename table
    op.rename_table("embeddings", "fact_embeddings")

    # Recreate indexes with new names
    op.create_index("idx_fact_embedding_entity", "fact_embeddings", ["entity_type", "entity_id"])
    op.create_index("idx_fact_embedding_model", "fact_embeddings", ["model_name"])

    # Recreate HNSW index with new table name
    op.execute(
        """
        CREATE INDEX idx_fact_embeddings_vector_1536 ON fact_embeddings
        USING hnsw ((embedding::vector(1536)) vector_cosine_ops)
        WHERE dimensions = 1536
        """
    )

    # Rename the unique constraint
    op.drop_constraint("uq_embedding_entity_model", "fact_embeddings", type_="unique")
    op.create_unique_constraint(
        "uq_fact_embedding_entity_model",
        "fact_embeddings",
        ["entity_type", "entity_id", "model_name"],
    )


def downgrade() -> None:  # cq: exempt
    # =========================================================================
    # Rename fact_embeddings -> embeddings
    # =========================================================================

    op.execute("DROP INDEX IF EXISTS idx_fact_embeddings_vector_1536")

    op.drop_index("idx_fact_embedding_entity", table_name="fact_embeddings")
    op.drop_index("idx_fact_embedding_model", table_name="fact_embeddings")

    op.rename_table("fact_embeddings", "embeddings")

    op.create_index("idx_embedding_entity", "embeddings", ["entity_type", "entity_id"])
    op.create_index("idx_embedding_model", "embeddings", ["model_name"])

    op.execute(
        """
        CREATE INDEX idx_embeddings_vector_1536 ON embeddings
        USING hnsw ((embedding::vector(1536)) vector_cosine_ops)
        WHERE dimensions = 1536
        """
    )

    op.drop_constraint("uq_fact_embedding_entity_model", "embeddings", type_="unique")
    op.create_unique_constraint(
        "uq_embedding_entity_model",
        "embeddings",
        ["entity_type", "entity_id", "model_name"],
    )

    # =========================================================================
    # Rename conv_sessions -> sessions
    # =========================================================================

    op.drop_index("idx_conv_sessions_session_id", table_name="conv_sessions")
    op.drop_index("idx_conv_sessions_updated_at", table_name="conv_sessions")

    op.rename_table("conv_sessions", "sessions")

    op.create_index("idx_sessions_session_id", "sessions", ["session_id"])
    op.create_index("idx_sessions_updated_at", "sessions", ["updated_at"])

    op.drop_constraint("uq_conv_sessions_session_id", "sessions", type_="unique")
    op.create_unique_constraint("uq_sessions_session_id", "sessions", ["session_id"])
