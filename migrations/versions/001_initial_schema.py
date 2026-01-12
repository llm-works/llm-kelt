"""Initial schema with all Phase 1 tables.

Revision ID: 001
Revises:
Create Date: 2025-01-05

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:  # cq: exempt
    # Enable pgvector extension for embedding storage
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # Content table - stores ingested content for reference and training
    op.create_table(
        "content",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("external_id", sa.String(length=255), nullable=True),
        sa.Column("source", sa.String(length=100), nullable=False),
        sa.Column("url", sa.Text(), nullable=True),
        sa.Column("title", sa.Text(), nullable=True),
        sa.Column("content_text", sa.Text(), nullable=True),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("embedding", sa.LargeBinary(), nullable=True),  # Vector stored as binary
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "fetched_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("content_hash"),
    )
    op.create_index("idx_content_source", "content", ["source"])
    op.create_index("idx_content_created", "content", ["created_at"])
    op.create_index("idx_content_external_id", "content", ["external_id"])

    # Add vector column separately (requires pgvector)
    op.execute("ALTER TABLE content DROP COLUMN embedding")
    op.execute("ALTER TABLE content ADD COLUMN embedding vector(1536)")

    # Feedback table - explicit user feedback on content
    op.create_table(
        "feedback",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("content_id", sa.BigInteger(), nullable=True),
        sa.Column("signal", sa.String(length=20), nullable=False),
        sa.Column("strength", sa.Float(), server_default="1.0", nullable=False),
        sa.Column("tags", postgresql.ARRAY(sa.String(50)), nullable=True),
        sa.Column("comment", sa.Text(), nullable=True),
        sa.Column("context", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["content_id"], ["content.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_feedback_signal", "feedback", ["signal"])
    op.create_index("idx_feedback_created", "feedback", ["created_at"])
    op.create_index("idx_feedback_content_id", "feedback", ["content_id"])

    # Preference pairs table - for DPO training
    op.create_table(
        "preference_pairs",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("context", sa.Text(), nullable=False),
        sa.Column("chosen", sa.Text(), nullable=False),
        sa.Column("rejected", sa.Text(), nullable=False),
        sa.Column("margin", sa.Float(), nullable=True),
        sa.Column("domain", sa.String(length=100), nullable=True),
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_preference_pairs_domain", "preference_pairs", ["domain"])
    op.create_index("idx_preference_pairs_created", "preference_pairs", ["created_at"])

    # Interactions table - implicit interaction signals
    op.create_table(
        "interactions",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("content_id", sa.BigInteger(), nullable=True),
        sa.Column("interaction_type", sa.String(length=50), nullable=False),
        sa.Column("duration_ms", sa.Integer(), nullable=True),
        sa.Column("scroll_depth", sa.Float(), nullable=True),
        sa.Column("context", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["content_id"], ["content.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_interactions_content", "interactions", ["content_id"])
    op.create_index("idx_interactions_type", "interactions", ["interaction_type"])
    op.create_index("idx_interactions_created", "interactions", ["created_at"])

    # Predictions table - hypothesis tracking for calibration
    op.create_table(
        "predictions",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("hypothesis", sa.Text(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("confidence_reasoning", sa.Text(), nullable=True),
        sa.Column("resolution_type", sa.String(length=50), nullable=True),
        sa.Column("resolution_date", sa.Date(), nullable=True),
        sa.Column("resolution_event", sa.Text(), nullable=True),
        sa.Column("resolution_metric", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("verification_source", sa.String(length=100), nullable=True),
        sa.Column("verification_url", sa.Text(), nullable=True),
        sa.Column("status", sa.String(length=20), server_default="pending", nullable=False),
        sa.Column("outcome", sa.String(length=20), nullable=True),
        sa.Column("outcome_confidence", sa.Float(), nullable=True),
        sa.Column("actual_result", sa.Text(), nullable=True),
        sa.Column("domain", sa.String(length=100), nullable=True),
        sa.Column("tags", postgresql.ARRAY(sa.String(50)), nullable=True),
        sa.Column("related_content_ids", postgresql.ARRAY(sa.BigInteger()), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_predictions_status", "predictions", ["status"])
    op.create_index("idx_predictions_domain", "predictions", ["domain"])
    op.create_index("idx_predictions_resolution_date", "predictions", ["resolution_date"])
    op.create_index("idx_predictions_created", "predictions", ["created_at"])

    # Directives table - standing user directives/goals
    op.create_table(
        "directives",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("directive_text", sa.Text(), nullable=False),
        sa.Column("directive_type", sa.String(length=50), nullable=True),
        sa.Column("parsed_rules", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("status", sa.String(length=20), server_default="active", nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_directives_status", "directives", ["status"])
    op.create_index("idx_directives_created", "directives", ["created_at"])

    # Create vector index for content embeddings (IVFFlat for approximate search)
    # Note: This requires a minimum number of rows to work well
    op.execute(
        """
        CREATE INDEX idx_content_embedding ON content
        USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100)
        """
    )


def downgrade() -> None:
    # Drop vector index first
    op.execute("DROP INDEX IF EXISTS idx_content_embedding")

    # Drop tables in reverse order (respecting foreign keys)
    op.drop_table("directives")
    op.drop_table("predictions")
    op.drop_table("interactions")
    op.drop_table("preference_pairs")
    op.drop_table("feedback")
    op.drop_table("content")

    # Drop pgvector extension (optional - may affect other databases)
    # op.execute("DROP EXTENSION IF EXISTS vector")
