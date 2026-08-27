# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-kelt Authors

"""Add sessions table for conversation persistence.

Revision ID: 002
Revises: 001
Create Date: 2026-04-01

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "002"
down_revision: str = "001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "sessions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("session_id", sa.String(255), nullable=False),
        sa.Column("messages", postgresql.JSONB(), nullable=False, server_default="[]"),
        sa.Column("token_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("config", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column("metadata", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("session_id", name="uq_sessions_session_id"),
    )
    op.create_index("idx_sessions_session_id", "sessions", ["session_id"])
    op.create_index("idx_sessions_updated_at", "sessions", ["updated_at"])


def downgrade() -> None:
    op.drop_index("idx_sessions_updated_at", table_name="sessions")
    op.drop_index("idx_sessions_session_id", table_name="sessions")
    op.drop_table("sessions")
