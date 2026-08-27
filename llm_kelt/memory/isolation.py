# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-kelt Authors

"""Isolation context for tenant boundaries."""

from dataclasses import dataclass
from typing import Any


def glob_to_like(context_key: str) -> tuple[str, bool]:
    """
    Convert a glob-style context key to a SQL LIKE pattern.

    Returns:
        (pattern, is_glob) — the LIKE pattern string and whether glob
        wildcards were present. The escape character is always backslash.
    """
    if "*" not in context_key and "?" not in context_key:
        return (context_key, False)

    # Escape the LIKE escape character first, then SQL wildcards, then translate glob
    pattern = context_key.replace("\\", "\\\\")
    pattern = pattern.replace("%", r"\%").replace("_", r"\_")
    pattern = pattern.replace("*", "%").replace("?", "_")
    return (pattern, True)


def build_context_filter(context_key: str | None, column: Any) -> Any:
    """
    Build SQLAlchemy filter for context_key with glob pattern support.

    Glob patterns:
        * - matches zero or more characters (like SQL %)
        ? - matches exactly one character (like SQL _)

    Examples:
        "acme_prod"     -> exact match only
        "acme_*"        -> matches "acme_prod", "acme_dev", etc.
        "acme_???"      -> matches "acme_dev", "acme_123" (exactly 3 chars)

    Args:
        context_key: Context key with optional glob wildcards (* or ?)
        column: SQLAlchemy column to filter on

    Returns:
        SQLAlchemy filter expression, or None if context_key is None
    """
    if context_key is None:
        return None

    pattern, is_glob = glob_to_like(context_key)
    if is_glob:
        return column.like(pattern, escape="\\")
    return column == context_key


@dataclass
class ClientContext:
    """
    Instructions for where to place data and how to partition queries.

    Pure data container - no business logic, no validation.
    The caller (e.g., llm-agent) is responsible for proper isolation.

    Attributes:
        context_key: Partition key (any string format). If None, no filtering applied.
            Supports glob patterns (* and ?) for hierarchical prefix matching:
            - "acme:prod:reviewer" - exact match (single agent)
            - "acme:prod:*" - prefix match (all agents in environment)
            - "acme:*" - prefix match (all environments for customer)
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
    # Supports glob patterns (* and ?) for prefix matching
    context_key: str | None = None

    # Schema where data lives - if None, defaults to "public"
    schema_name: str | None = None
