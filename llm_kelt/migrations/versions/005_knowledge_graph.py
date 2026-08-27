# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-kelt Authors

"""Add Knowledge Graph tables and rename metadata columns.

Renames metadata → extra in existing tables:
- atomic_facts.metadata → atomic_facts.extra
- atomic_preference_details.metadata → atomic_preference_details.extra
- contents.metadata → contents.extra
- conv_sessions.metadata → conv_sessions.extra
- fact_relationships.metadata → fact_relationships.extra

Creates entity-centric knowledge management with scoped subgraphs:
- kg_entities: Canonical entities with identity-based dedup
- kg_entity_aliases: Alias→entity mapping for resolution
- kg_entity_relationships: Entity→entity edges
- kg_fact_entities: Fact→entity linkage
- kg_entity_refs: Reference tracking for provenance/signals

Revision ID: 005
Revises: 004
Create Date: 2026-05-16
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "005"
down_revision: str = "004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:  # cq: exempt
    # =========================================================================
    # Rename metadata → extra in existing tables
    # =========================================================================
    op.alter_column("atomic_facts", "metadata", new_column_name="extra")
    op.alter_column("atomic_preference_details", "metadata", new_column_name="extra")
    op.alter_column("contents", "metadata", new_column_name="extra")
    op.alter_column("conv_sessions", "metadata", new_column_name="extra")
    op.alter_column("fact_relationships", "metadata", new_column_name="extra")

    # =========================================================================
    # kg_entities
    # =========================================================================
    op.create_table(
        "kg_entities",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("scope_key", sa.String(length=255), nullable=False),
        sa.Column("canonical_name", sa.String(length=255), nullable=False),
        sa.Column("entity_type", sa.String(length=50), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("extra", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "scope_key", "canonical_name", "entity_type", name="uq_kg_entity_identity"
        ),
    )
    op.create_index("ix_kg_entities_scope", "kg_entities", ["scope_key"])
    op.create_index("ix_kg_entities_scope_type", "kg_entities", ["scope_key", "entity_type"])
    op.create_index(
        "ix_kg_entities_scope_prefix",
        "kg_entities",
        ["scope_key"],
        postgresql_ops={"scope_key": "varchar_pattern_ops"},
    )

    # =========================================================================
    # kg_entity_aliases
    # =========================================================================
    op.create_table(
        "kg_entity_aliases",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("entity_id", sa.BigInteger(), nullable=False),
        sa.Column("scope_key", sa.String(length=255), nullable=False),
        sa.Column("alias", sa.String(length=255), nullable=False),
        sa.Column("alias_normalized", sa.String(length=255), nullable=False),
        sa.Column("extra", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.ForeignKeyConstraint(["entity_id"], ["kg_entities.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("scope_key", "alias_normalized", name="uq_kg_alias_identity"),
    )
    op.create_index("ix_kg_entity_aliases_scope", "kg_entity_aliases", ["scope_key"])
    op.create_index("ix_kg_entity_aliases_entity", "kg_entity_aliases", ["entity_id"])
    op.create_index(
        "ix_kg_entity_aliases_scope_prefix",
        "kg_entity_aliases",
        ["scope_key"],
        postgresql_ops={"scope_key": "varchar_pattern_ops"},
    )

    # =========================================================================
    # kg_entity_relationships
    # =========================================================================
    op.create_table(
        "kg_entity_relationships",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("scope_key", sa.String(length=255), nullable=False),
        sa.Column("from_entity_id", sa.BigInteger(), nullable=False),
        sa.Column("to_entity_id", sa.BigInteger(), nullable=False),
        sa.Column("relationship_type", sa.String(length=50), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False, server_default="1.0"),
        sa.Column("extra", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.ForeignKeyConstraint(["from_entity_id"], ["kg_entities.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["to_entity_id"], ["kg_entities.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "scope_key",
            "from_entity_id",
            "to_entity_id",
            "relationship_type",
            name="uq_kg_entity_rel",
        ),
    )
    op.create_index("ix_kg_entity_relationships_scope", "kg_entity_relationships", ["scope_key"])
    op.create_index(
        "ix_kg_entity_relationships_from", "kg_entity_relationships", ["from_entity_id"]
    )
    op.create_index("ix_kg_entity_relationships_to", "kg_entity_relationships", ["to_entity_id"])
    op.create_index(
        "ix_kg_entity_relationships_type",
        "kg_entity_relationships",
        ["from_entity_id", "relationship_type"],
    )
    op.create_index(
        "ix_kg_entity_relationships_scope_prefix",
        "kg_entity_relationships",
        ["scope_key"],
        postgresql_ops={"scope_key": "varchar_pattern_ops"},
    )

    # =========================================================================
    # kg_fact_entities
    # =========================================================================
    op.create_table(
        "kg_fact_entities",
        sa.Column("fact_id", sa.BigInteger(), nullable=False),
        sa.Column("entity_id", sa.BigInteger(), nullable=False),
        sa.Column("scope_key", sa.String(length=255), nullable=False),
        sa.Column("role", sa.String(length=50), nullable=False, server_default="subject"),
        sa.Column("confidence", sa.Float(), nullable=False, server_default="1.0"),
        sa.Column("extra", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.ForeignKeyConstraint(["fact_id"], ["atomic_facts.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["entity_id"], ["kg_entities.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("fact_id", "entity_id"),
    )
    op.create_index("ix_kg_fact_entities_scope", "kg_fact_entities", ["scope_key"])
    op.create_index("ix_kg_fact_entities_entity", "kg_fact_entities", ["entity_id"])
    op.create_index("ix_kg_fact_entities_fact", "kg_fact_entities", ["fact_id"])
    op.create_index(
        "ix_kg_fact_entities_scope_prefix",
        "kg_fact_entities",
        ["scope_key"],
        postgresql_ops={"scope_key": "varchar_pattern_ops"},
    )

    # =========================================================================
    # kg_entity_refs
    # =========================================================================
    op.create_table(
        "kg_entity_refs",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("entity_id", sa.BigInteger(), nullable=False),
        sa.Column("scope_key", sa.String(length=255), nullable=False),
        sa.Column("source_type", sa.String(length=50), nullable=False),
        sa.Column("source_id", sa.String(length=255), nullable=True),
        sa.Column("source_url", sa.String(length=2048), nullable=True),
        sa.Column("snippet", sa.Text(), nullable=True),
        sa.Column("sentiment", sa.Float(), nullable=True),
        sa.Column("extra", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column(
            "ref_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.ForeignKeyConstraint(["entity_id"], ["kg_entities.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_kg_entity_refs_scope", "kg_entity_refs", ["scope_key"])
    op.create_index("ix_kg_entity_refs_entity", "kg_entity_refs", ["entity_id"])
    op.create_index("ix_kg_entity_refs_entity_time", "kg_entity_refs", ["entity_id", "ref_at"])
    op.create_index(
        "ix_kg_entity_refs_scope_prefix",
        "kg_entity_refs",
        ["scope_key"],
        postgresql_ops={"scope_key": "varchar_pattern_ops"},
    )


def downgrade() -> None:
    op.drop_table("kg_entity_refs")
    op.drop_table("kg_fact_entities")
    op.drop_table("kg_entity_relationships")
    op.drop_table("kg_entity_aliases")
    op.drop_table("kg_entities")

    # Rename extra → metadata in existing tables
    op.alter_column("fact_relationships", "extra", new_column_name="metadata")
    op.alter_column("conv_sessions", "extra", new_column_name="metadata")
    op.alter_column("contents", "extra", new_column_name="metadata")
    op.alter_column("atomic_preference_details", "extra", new_column_name="metadata")
    op.alter_column("atomic_facts", "extra", new_column_name="metadata")
