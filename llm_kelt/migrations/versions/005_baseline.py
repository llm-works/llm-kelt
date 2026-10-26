# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-kelt Authors

"""Oldest revision a schema upgrades from.

SchemaManager.ensure_schema() creates a new schema from the models and stamps it with
the head revision; migrations only upgrade existing schemas. A schema at this revision
has the layout the models defined at 005 (the knowledge graph tables included), so the
revision has no steps of its own and the chain cannot build a schema from empty.

Revision ID: 005
Revises:
Create Date: 2026-05-16
"""

from collections.abc import Sequence

revision: str = "005"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    raise RuntimeError(
        "migrations upgrade existing schemas only; "
        "create a new schema with SchemaManager.ensure_schema()"
    )


def downgrade() -> None:
    raise RuntimeError("005 is the oldest revision; there is no earlier schema to return to")
