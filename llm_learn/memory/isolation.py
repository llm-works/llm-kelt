"""Isolation context for tenant boundaries."""

from dataclasses import dataclass


@dataclass
class IsolationContext:
    """
    Instructions for where to place data and how to partition queries.

    Pure data container - no business logic, no validation.
    The caller (e.g., llm-agent) is responsible for proper isolation.

    Attributes:
        context_key: Partition key (any string format). If None, no filtering applied.
            Supports SQL LIKE patterns (% and _) for hierarchical prefix matching:
            - "acme:prod:reviewer" - exact match (single profile)
            - "acme:prod:%" - prefix match (all profiles in workspace)
            - "acme:%" - prefix match (all workspaces in domain)
        schema_name: Schema where data lives. If None, defaults to "public".

    Examples:
        # Exact match - single profile
        context = IsolationContext(
            context_key="acme:prod:reviewer",
            schema_name="customer_acme"
        )

        # Prefix match - all profiles in a workspace
        context = IsolationContext(context_key="acme:prod:%")

        # Prefix match - all workspaces in a domain
        context = IsolationContext(context_key="acme:%")

        # Single tenant: no filtering, public schema (simplest)
        context = IsolationContext()

    Hierarchical Partitioning:
        Use colon-separated (or any delimiter) keys for hierarchy:
        - domain:workspace:profile (e.g., "acme:prod:reviewer")
        - customer:environment:agent (e.g., "acme:staging:summarizer")

        Then query at any level using wildcards:
        - All profiles: "acme:prod:%"
        - All workspaces: "acme:%"
        - All data: "%"

    Responsibility:
        llm-learn is a passthrough - it applies the context as given.
        The caller is responsible for security and proper isolation.
    """

    # Partition key - if None, no filtering (caller owns all data)
    # Supports SQL LIKE patterns (% and _) for prefix matching
    context_key: str | None = None

    # Schema where data lives - if None, defaults to "public"
    schema_name: str | None = None
