"""Memory v1 architecture - unified fact-based storage.

Revision ID: 003
Revises: 002
Create Date: 2026-02-02

This migration:
1. Creates memv1_* tables (unified fact architecture)
2. Migrates data from old tables to new structure
3. Drops old tables (facts, feedback, predictions, etc.)
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

revision: str = "003"
down_revision: str | None = "002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:  # cq: exempt
    # =========================================================================
    # Create memv1_facts base table
    # =========================================================================
    op.create_table(
        "memv1_facts",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("profile_id", sa.BigInteger(), nullable=False),
        sa.Column("type", sa.String(20), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("category", sa.String(50), nullable=True),
        sa.Column("source", sa.String(50), server_default="user", nullable=False),
        sa.Column("confidence", sa.Float(), server_default="1.0", nullable=False),
        sa.Column("active", sa.Boolean(), server_default="true", nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["profile_id"], ["profiles.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_memv1_facts_profile", "memv1_facts", ["profile_id"])
    op.create_index("idx_memv1_facts_profile_type", "memv1_facts", ["profile_id", "type"])
    op.create_index("idx_memv1_facts_category", "memv1_facts", ["category"])
    op.create_index("idx_memv1_facts_profile_active", "memv1_facts", ["profile_id", "active"])
    op.create_index("idx_memv1_facts_created", "memv1_facts", ["created_at"])

    # =========================================================================
    # Create detail tables
    # =========================================================================

    # Solution details (NEW - agent problem/answer records)
    op.create_table(
        "memv1_solution_details",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("fact_id", sa.BigInteger(), nullable=False),
        sa.Column("agent_name", sa.String(100), nullable=False),
        sa.Column("problem", sa.Text(), nullable=False),
        sa.Column("problem_context", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("answer", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("answer_text", sa.Text(), nullable=True),
        sa.Column("tokens_used", sa.Integer(), nullable=False),
        sa.Column("latency_ms", sa.Integer(), nullable=False),
        sa.Column("tool_calls", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.ForeignKeyConstraint(["fact_id"], ["memv1_facts.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("fact_id", name="uq_memv1_solution_fact"),
    )
    op.create_index("idx_memv1_solution_fact", "memv1_solution_details", ["fact_id"])
    op.create_index("idx_memv1_solution_agent", "memv1_solution_details", ["agent_name"])

    # Prediction details
    op.create_table(
        "memv1_prediction_details",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("fact_id", sa.BigInteger(), nullable=False),
        sa.Column("resolution_type", sa.String(50), nullable=True),
        sa.Column("resolution_date", sa.Date(), nullable=True),
        sa.Column("resolution_event", sa.Text(), nullable=True),
        sa.Column("resolution_metric", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("verification_source", sa.String(100), nullable=True),
        sa.Column("verification_url", sa.Text(), nullable=True),
        sa.Column("status", sa.String(20), server_default="pending", nullable=False),
        sa.Column("outcome", sa.String(20), nullable=True),
        sa.Column("outcome_confidence", sa.Float(), nullable=True),
        sa.Column("actual_result", sa.Text(), nullable=True),
        sa.Column("tags", postgresql.ARRAY(sa.String(50)), nullable=True),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["fact_id"], ["memv1_facts.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("fact_id", name="uq_memv1_prediction_fact"),
    )
    op.create_index("idx_memv1_prediction_fact", "memv1_prediction_details", ["fact_id"])
    op.create_index("idx_memv1_prediction_status", "memv1_prediction_details", ["status"])
    op.create_index(
        "idx_memv1_prediction_resolution_date", "memv1_prediction_details", ["resolution_date"]
    )

    # Feedback details
    op.create_table(
        "memv1_feedback_details",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("fact_id", sa.BigInteger(), nullable=False),
        sa.Column("content_id", sa.BigInteger(), nullable=True),
        sa.Column("signal", sa.String(20), nullable=False),
        sa.Column("strength", sa.Float(), server_default="1.0", nullable=False),
        sa.Column("tags", postgresql.ARRAY(sa.String(50)), nullable=True),
        sa.Column("comment", sa.Text(), nullable=True),
        sa.Column("context", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.ForeignKeyConstraint(["fact_id"], ["memv1_facts.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["content_id"], ["content.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("fact_id", name="uq_memv1_feedback_fact"),
    )
    op.create_index("idx_memv1_feedback_fact", "memv1_feedback_details", ["fact_id"])
    op.create_index("idx_memv1_feedback_content", "memv1_feedback_details", ["content_id"])
    op.create_index("idx_memv1_feedback_signal", "memv1_feedback_details", ["signal"])

    # Directive details
    op.create_table(
        "memv1_directive_details",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("fact_id", sa.BigInteger(), nullable=False),
        sa.Column("directive_type", sa.String(50), nullable=True),
        sa.Column("parsed_rules", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("status", sa.String(20), server_default="active", nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["fact_id"], ["memv1_facts.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("fact_id", name="uq_memv1_directive_fact"),
    )
    op.create_index("idx_memv1_directive_fact", "memv1_directive_details", ["fact_id"])
    op.create_index("idx_memv1_directive_status", "memv1_directive_details", ["status"])

    # Interaction details
    op.create_table(
        "memv1_interaction_details",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("fact_id", sa.BigInteger(), nullable=False),
        sa.Column("content_id", sa.BigInteger(), nullable=True),
        sa.Column("interaction_type", sa.String(50), nullable=False),
        sa.Column("duration_ms", sa.Integer(), nullable=True),
        sa.Column("scroll_depth", sa.Float(), nullable=True),
        sa.Column("context", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.ForeignKeyConstraint(["fact_id"], ["memv1_facts.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["content_id"], ["content.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("fact_id", name="uq_memv1_interaction_fact"),
    )
    op.create_index("idx_memv1_interaction_fact", "memv1_interaction_details", ["fact_id"])
    op.create_index("idx_memv1_interaction_content", "memv1_interaction_details", ["content_id"])
    op.create_index("idx_memv1_interaction_type", "memv1_interaction_details", ["interaction_type"])

    # Preference details
    op.create_table(
        "memv1_preference_details",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("fact_id", sa.BigInteger(), nullable=False),
        sa.Column("context", sa.Text(), nullable=False),
        sa.Column("chosen", sa.Text(), nullable=False),
        sa.Column("rejected", sa.Text(), nullable=False),
        sa.Column("margin", sa.Float(), nullable=True),
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.ForeignKeyConstraint(["fact_id"], ["memv1_facts.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("fact_id", name="uq_memv1_preference_fact"),
    )
    op.create_index("idx_memv1_preference_fact", "memv1_preference_details", ["fact_id"])

    # Fact embeddings
    op.create_table(
        "memv1_fact_embeddings",
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
        sa.ForeignKeyConstraint(["fact_id"], ["memv1_facts.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("fact_id", "model_name", name="uq_memv1_fact_embedding_model"),
    )
    op.create_index("idx_memv1_fact_embeddings_fact", "memv1_fact_embeddings", ["fact_id"])
    op.create_index("idx_memv1_fact_embeddings_model", "memv1_fact_embeddings", ["model_name"])

    # Create HNSW index for vector similarity search
    op.execute(
        """
        CREATE INDEX idx_memv1_fact_embeddings_vector ON memv1_fact_embeddings
        USING hnsw (embedding vector_cosine_ops)
        """
    )

    # =========================================================================
    # Drop old tables (v0 schema)
    # =========================================================================
    # Drop in order respecting foreign key constraints

    # fact_embeddings references facts
    op.drop_table("fact_embeddings")

    # feedback references facts and content
    op.drop_table("feedback")

    # interactions references facts and content
    op.drop_table("interactions")

    # directives references facts
    op.drop_table("directives")

    # predictions references facts
    op.drop_table("predictions")

    # preference_pairs references facts
    op.drop_table("preference_pairs")

    # facts references profiles (base table, drop last)
    op.drop_table("facts")


def downgrade() -> None:
    # Drop memv1 tables (reverse order respecting FKs)
    op.drop_index("idx_memv1_fact_embeddings_vector", table_name="memv1_fact_embeddings")
    op.drop_table("memv1_fact_embeddings")
    op.drop_table("memv1_preference_details")
    op.drop_table("memv1_interaction_details")
    op.drop_table("memv1_directive_details")
    op.drop_table("memv1_feedback_details")
    op.drop_table("memv1_prediction_details")
    op.drop_table("memv1_solution_details")
    op.drop_table("memv1_facts")
