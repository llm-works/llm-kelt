"""Initial schema - all tables for atomic memory architecture.

Revision ID: 001
Revises:
Create Date: 2025-01-05

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

revision: str = "001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:  # cq: exempt
    # Enable pgvector extension
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # =========================================================================
    # Content table (raw ingested content)
    # =========================================================================

    op.create_table(
        "content",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("context_key", sa.String(255), nullable=True),
        sa.Column("external_id", sa.String(255), nullable=True),
        sa.Column("source", sa.String(100), nullable=False),
        sa.Column("url", sa.Text(), nullable=True),
        sa.Column("title", sa.Text(), nullable=True),
        sa.Column("content_text", sa.Text(), nullable=True),
        sa.Column("content_hash", sa.String(64), nullable=False),
        sa.Column("metadata", postgresql.JSONB(), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "fetched_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("context_key", "content_hash", name="uq_content_context_hash"),
    )
    op.create_index("idx_content_context", "content", ["context_key"])
    op.create_index("idx_content_source", "content", ["source"])
    op.create_index("idx_content_created", "content", ["created_at"])
    op.create_index("idx_content_external_id", "content", ["external_id"])
    # Prefix index for efficient LIKE queries (pattern matching)
    op.create_index(
        "idx_content_context_prefix",
        "content",
        ["context_key"],
        postgresql_ops={"context_key": "text_pattern_ops"},
    )
    # Partial unique index for NULL context_key to ensure deduplication works
    # When context_key IS NULL, content_hash must be unique (global scope deduplication)
    op.create_index(
        "uq_content_null_context_hash",
        "content",
        ["content_hash"],
        unique=True,
        postgresql_where=sa.text("context_key IS NULL"),
    )

    # =========================================================================
    # Embeddings table (entity-type agnostic vector storage)
    # =========================================================================

    op.create_table(
        "embeddings",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("entity_type", sa.String(50), nullable=False),
        sa.Column("entity_id", sa.String(64), nullable=False),
        sa.Column("model_name", sa.String(100), nullable=False),
        sa.Column("dimensions", sa.Integer(), nullable=False),
        sa.Column("embedding", Vector(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "entity_type", "entity_id", "model_name", name="uq_embedding_entity_model"
        ),
    )
    op.create_index("idx_embedding_entity", "embeddings", ["entity_type", "entity_id"])
    op.create_index("idx_embedding_model", "embeddings", ["model_name"])

    # HNSW index for vector similarity search
    op.execute(
        """
        CREATE INDEX idx_embeddings_vector ON embeddings
        USING hnsw (embedding vector_cosine_ops)
        """
    )

    # =========================================================================
    # Atomic facts base table
    # =========================================================================

    op.create_table(
        "atomic_facts",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("context_key", sa.String(255), nullable=True),
        sa.Column("type", sa.String(20), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("category", sa.String(50), nullable=True),
        sa.Column("source", sa.String(50), server_default="user", nullable=False),
        sa.Column("confidence", sa.Float(), server_default="1.0", nullable=False),
        sa.Column("active", sa.Boolean(), server_default="true", nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_atomic_facts_context", "atomic_facts", ["context_key"])
    op.create_index("idx_atomic_facts_context_type", "atomic_facts", ["context_key", "type"])
    op.create_index("idx_atomic_facts_category", "atomic_facts", ["category"])
    op.create_index("idx_atomic_facts_context_active", "atomic_facts", ["context_key", "active"])
    op.create_index("idx_atomic_facts_created", "atomic_facts", ["created_at"])
    # Prefix index for efficient LIKE queries (pattern matching)
    op.create_index(
        "idx_atomic_facts_context_prefix",
        "atomic_facts",
        ["context_key"],
        postgresql_ops={"context_key": "text_pattern_ops"},
    )

    # =========================================================================
    # Atomic fact detail tables
    # =========================================================================

    op.create_table(
        "atomic_solution_details",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("fact_id", sa.BigInteger(), nullable=False),
        sa.Column("agent_name", sa.String(100), nullable=False),
        sa.Column("problem", sa.Text(), nullable=False),
        sa.Column("problem_context", postgresql.JSONB(), nullable=False),
        sa.Column("answer", postgresql.JSONB(), nullable=False),
        sa.Column("answer_text", sa.Text(), nullable=True),
        sa.Column("tokens_used", sa.Integer(), nullable=False),
        sa.Column("latency_ms", sa.Integer(), nullable=False),
        sa.Column("tool_calls", postgresql.JSONB(), nullable=True),
        sa.ForeignKeyConstraint(["fact_id"], ["atomic_facts.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("fact_id", name="uq_atomic_solution_fact"),
    )
    op.create_index("idx_atomic_solution_fact", "atomic_solution_details", ["fact_id"])
    op.create_index("idx_atomic_solution_agent", "atomic_solution_details", ["agent_name"])

    op.create_table(
        "atomic_prediction_details",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("fact_id", sa.BigInteger(), nullable=False),
        sa.Column("resolution_type", sa.String(50), nullable=True),
        sa.Column("resolution_date", sa.Date(), nullable=True),
        sa.Column("resolution_event", sa.Text(), nullable=True),
        sa.Column("resolution_metric", postgresql.JSONB(), nullable=True),
        sa.Column("verification_source", sa.String(100), nullable=True),
        sa.Column("verification_url", sa.Text(), nullable=True),
        sa.Column("status", sa.String(20), server_default="pending", nullable=False),
        sa.Column("outcome", sa.String(20), nullable=True),
        sa.Column("outcome_confidence", sa.Float(), nullable=True),
        sa.Column("actual_result", sa.Text(), nullable=True),
        sa.Column("tags", postgresql.ARRAY(sa.String(50)), nullable=True),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["fact_id"], ["atomic_facts.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("fact_id", name="uq_atomic_prediction_fact"),
    )
    op.create_index("idx_atomic_prediction_fact", "atomic_prediction_details", ["fact_id"])
    op.create_index("idx_atomic_prediction_status", "atomic_prediction_details", ["status"])
    op.create_index(
        "idx_atomic_prediction_resolution_date", "atomic_prediction_details", ["resolution_date"]
    )

    op.create_table(
        "atomic_feedback_details",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("fact_id", sa.BigInteger(), nullable=False),
        sa.Column("content_id", sa.BigInteger(), nullable=True),
        sa.Column("signal", sa.String(20), nullable=False),
        sa.Column("strength", sa.Float(), server_default="1.0", nullable=False),
        sa.Column("tags", postgresql.ARRAY(sa.String(50)), nullable=True),
        sa.Column("comment", sa.Text(), nullable=True),
        sa.Column("context", postgresql.JSONB(), nullable=True),
        sa.ForeignKeyConstraint(["fact_id"], ["atomic_facts.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["content_id"], ["content.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("fact_id", name="uq_atomic_feedback_fact"),
    )
    op.create_index("idx_atomic_feedback_fact", "atomic_feedback_details", ["fact_id"])
    op.create_index("idx_atomic_feedback_content", "atomic_feedback_details", ["content_id"])
    op.create_index("idx_atomic_feedback_signal", "atomic_feedback_details", ["signal"])

    op.create_table(
        "atomic_directive_details",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("fact_id", sa.BigInteger(), nullable=False),
        sa.Column("directive_type", sa.String(50), nullable=True),
        sa.Column("parsed_rules", postgresql.JSONB(), nullable=True),
        sa.Column("status", sa.String(20), server_default="active", nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["fact_id"], ["atomic_facts.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("fact_id", name="uq_atomic_directive_fact"),
    )
    op.create_index("idx_atomic_directive_fact", "atomic_directive_details", ["fact_id"])
    op.create_index("idx_atomic_directive_status", "atomic_directive_details", ["status"])

    op.create_table(
        "atomic_interaction_details",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("fact_id", sa.BigInteger(), nullable=False),
        sa.Column("content_id", sa.BigInteger(), nullable=True),
        sa.Column("interaction_type", sa.String(50), nullable=False),
        sa.Column("duration_ms", sa.Integer(), nullable=True),
        sa.Column("scroll_depth", sa.Float(), nullable=True),
        sa.Column("context", postgresql.JSONB(), nullable=True),
        sa.ForeignKeyConstraint(["fact_id"], ["atomic_facts.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["content_id"], ["content.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("fact_id", name="uq_atomic_interaction_fact"),
    )
    op.create_index("idx_atomic_interaction_fact", "atomic_interaction_details", ["fact_id"])
    op.create_index("idx_atomic_interaction_content", "atomic_interaction_details", ["content_id"])
    op.create_index(
        "idx_atomic_interaction_type", "atomic_interaction_details", ["interaction_type"]
    )

    op.create_table(
        "atomic_preference_details",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("fact_id", sa.BigInteger(), nullable=False),
        sa.Column("context", sa.Text(), nullable=False),
        sa.Column("chosen", sa.Text(), nullable=False),
        sa.Column("rejected", sa.Text(), nullable=False),
        sa.Column("margin", sa.Float(), nullable=True),
        sa.Column("metadata", postgresql.JSONB(), nullable=True),
        sa.ForeignKeyConstraint(["fact_id"], ["atomic_facts.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("fact_id", name="uq_atomic_preference_fact"),
    )
    op.create_index("idx_atomic_preference_fact", "atomic_preference_details", ["fact_id"])


def downgrade() -> None:
    """Downgrade not supported - this is the initial schema.

    To reset the database, drop all tables and re-run upgrade.
    """
    raise NotImplementedError(
        "Downgrade from initial schema is not supported. "
        "To reset the database, drop all tables manually and re-run migrations."
    )
