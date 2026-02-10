"""Isolation context for tenant boundaries."""

from dataclasses import dataclass
from typing import Any


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

    # Glob-style pattern matching: * (zero or more), ? (exactly one)
    if "*" in context_key or "?" in context_key:
        # Escape SQL wildcards first, then translate glob to SQL LIKE
        pattern = context_key.replace("%", r"\%").replace("_", r"\_")
        pattern = pattern.replace("*", "%").replace("?", "_")
        return column.like(pattern, escape="\\")
    else:
        # Exact match - no wildcards
        return column == context_key


@dataclass
class IsolationContext:
    """
    Instructions for where to place data and how to partition queries.

    Pure data container - no business logic, no validation.
    The caller (e.g., llm-agent) is responsible for proper isolation.

    Attributes:
        context_key: Partition key (any string format). If None, no filtering applied.
            Supports glob patterns (* and ?) for hierarchical prefix matching:
            - "acme:prod:reviewer" - exact match (single profile)
            - "acme:prod:*" - prefix match (all profiles in workspace)
            - "acme:*" - prefix match (all workspaces in domain)
        schema_name: Schema where data lives. If None, defaults to "public".

    Examples:
        # Exact match - single profile
        context = IsolationContext(
            context_key="acme:prod:reviewer",
            schema_name="customer_acme"
        )

        # Prefix match - all profiles in a workspace
        context = IsolationContext(context_key="acme:prod:*")

        # Prefix match - all workspaces in a domain
        context = IsolationContext(context_key="acme:*")

        # Single tenant: no filtering, public schema (simplest)
        context = IsolationContext()

    Hierarchical Partitioning:
        Use colon-separated (or any delimiter) keys for hierarchy:
        - domain:workspace:profile (e.g., "acme:prod:reviewer")
        - customer:environment:agent (e.g., "acme:staging:summarizer")

        Then query at any level using glob wildcards:
        - All profiles: "acme:prod:*"
        - All workspaces: "acme:*"
        - All data: "*"

    Responsibility:
        llm-learn is a passthrough - it applies the context as given.
        The caller is responsible for security and proper isolation.
    """

    # Partition key - if None, no filtering (caller owns all data)
    # Supports glob patterns (* and ?) for prefix matching
    context_key: str | None = None

    # Schema where data lives - if None, defaults to "public"
    schema_name: str | None = None
