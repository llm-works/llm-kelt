# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-kelt Authors

"""Isolation context for tenant boundaries."""

from dataclasses import dataclass
from typing import Any

from sqlalchemy import ColumnElement, SQLColumnExpression

from ..core.keys import key_filter


def build_context_filter(
    context_key: str | None, column: SQLColumnExpression[Any]
) -> ColumnElement[bool] | None:
    """
    Build SQLAlchemy filter for a context key or pattern.

    Patterns (see llm_kelt.core.keys):
        *     - any run of characters
        ?     - exactly one character
        K:**  - K itself and every key below it

    Args:
        context_key: Key or pattern. None means no filtering.
        column: SQLAlchemy column to filter on

    Returns:
        SQLAlchemy filter expression, or None if context_key is None
    """
    if context_key is None:
        return None
    return key_filter(context_key, column)


@dataclass
class ClientContext:
    """
    Instructions for where to place data and how to partition queries.

    Pure data container - no business logic, no validation.
    The caller (e.g., llm-agent) is responsible for proper isolation.

    Attributes:
        context_key: Partition key (any string format). If None, no filtering applied.
            Reads accept a pattern; a client opened with a pattern reads across keys
            but cannot write.
            - "acme:prod:reviewer" - exact match (single agent)
            - "acme:prod:*" - every key below acme:prod
            - "acme:prod:**" - acme:prod itself and every key below it
        schema_name: Schema where data lives. If None, defaults to "public".

    Examples:
        # Exact match - single agent
        context = ClientContext(
            context_key="acme:prod:reviewer",
            schema_name="customer_acme"
        )

        # Prefix match - all agents in an environment
        context = ClientContext(context_key="acme:prod:*")

        # Prefix match - all environments for a customer
        context = ClientContext(context_key="acme:*")

        # Single tenant: no filtering, public schema (simplest)
        context = ClientContext()

    Hierarchical Partitioning:
        Use colon-separated (or any delimiter) keys for hierarchy:
        - customer:environment:agent (e.g., "acme:prod:reviewer")
        - tenant:project:instance (e.g., "acme:api:summarizer")

        Then query at any level using glob wildcards:
        - All agents: "acme:prod:*"
        - All environments: "acme:*"
        - All data: "*"

    Responsibility:
        llm-kelt is a passthrough - it applies the context as given.
        The caller is responsible for security and proper isolation.
    """

    # Partition key - if None, no filtering (caller owns all data)
    # Reads accept a pattern; writes need an exact key
    context_key: str | None = None

    # Schema where data lives - if None, defaults to "public"
    schema_name: str | None = None
