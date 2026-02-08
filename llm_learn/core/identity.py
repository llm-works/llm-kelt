"""Profile identity resolution - converts names to deterministic IDs.

This module provides tooling to manage the domain → workspace → profile hierarchy,
allowing applications to specify identities by name and optionally override with
explicit IDs for migrations/renaming scenarios.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .domain import Domain
from .exceptions import ValidationError
from .profile import Profile
from .workspace import Workspace


@dataclass(frozen=True)
class ProfileIdentity:
    """Type-safe resolved profile identity.

    Represents a profile's location in the domain/workspace/profile hierarchy
    with all IDs resolved and validated. This is the output of IdentityResolver.

    All ID fields are guaranteed to be valid 32-character hex hashes.
    """

    # Names (human-readable)
    name: str
    """Profile name/identifier (unique within workspace)."""

    workspace: str
    """Workspace name/identifier (unique within domain)."""

    domain: str | None
    """Domain name/identifier (null for single-tenant setups)."""

    # IDs (32-char hashes, immutable after resolution)
    profile_id: str
    """Profile ID hash (32 chars, guaranteed present)."""

    workspace_id: str
    """Workspace ID hash (32 chars, guaranteed present)."""

    domain_id: str | None
    """Domain ID hash (32 chars, only present if domain exists)."""

    def __repr__(self) -> str:
        """Human-readable representation."""
        path = f"{self.domain or 'null'}/{self.workspace}/{self.name}"
        return f"<ProfileIdentity({path!r}, profile_id={self.profile_id[:8]}...)>"


class IdentityResolver:
    """Resolves flexible dict/DotDict config to type-safe ProfileIdentity.

    Handles the full domain/workspace/profile hierarchy, generating deterministic
    IDs from names or using explicit IDs when provided (for migrations/renaming).

    Examples:
        # Simple: just name, defaults to default workspace
        config = {"name": "joke-teller"}
        identity = IdentityResolver.resolve(config)
        # → ProfileIdentity(name="joke-teller", workspace="default", ...)

        # Multi-tenant: specify full hierarchy
        config = {
            "domain": "acme",
            "workspace": "production",
            "name": "code-reviewer"
        }
        identity = IdentityResolver.resolve(config)
        # → IDs generated from full path

        # Migration: renamed agent, keep history
        config = {
            "name": "comedian",  # New name
            "profile_id": "6fae98ab8d170809de4e057cf7d718da"  # Old hash
        }
        identity = IdentityResolver.resolve(config)
        # → Uses explicit ID, name is just metadata

        # With defaults (e.g., agent name)
        config = {}  # Empty
        identity = IdentityResolver.resolve(config, defaults={"name": "my-agent"})
        # → Uses default name
    """

    @staticmethod
    def resolve(
        config: dict[str, Any] | None = None,
        defaults: dict[str, Any] | None = None,
    ) -> ProfileIdentity:
        """Resolve profile identity from flexible configuration.

        Args:
            config: Configuration dict (from YAML or API) with optional fields:
                - name: Profile name (required unless in defaults)
                - workspace: Workspace name (default: "default")
                - domain: Domain name (default: null)
                - profile_id: Explicit profile ID hash (optional, for migrations)
                - workspace_id: Explicit workspace ID hash (optional)
                - domain_id: Explicit domain ID hash (optional)
            defaults: Default values to use if not in config (e.g., {"name": "agent-x"})

        Returns:
            Type-safe ProfileIdentity with all IDs resolved.

        Raises:
            ValidationError: If required fields missing or IDs invalid.
        """
        config = config or {}
        defaults = defaults or {}

        # Extract and validate names
        name, workspace, domain = _extract_names(config, defaults)

        # Extract and validate explicit IDs
        explicit_ids = _extract_and_validate_explicit_ids(config)

        # Resolve final IDs
        domain_id, workspace_id, profile_id = _resolve_ids(name, workspace, domain, explicit_ids)

        return ProfileIdentity(
            name=name,
            workspace=workspace,
            domain=domain,
            profile_id=profile_id,
            workspace_id=workspace_id,
            domain_id=domain_id,
        )


def _extract_names(config: dict[str, Any], defaults: dict[str, Any]) -> tuple[str, str, str | None]:
    """Extract and validate profile/workspace/domain names from config.

    Returns:
        Tuple of (name, workspace, domain).

    Raises:
        ValidationError: If profile name is missing.
    """
    name = config.get("name") or defaults.get("name")
    if not name:
        raise ValidationError("Profile name is required (in config or defaults)")

    workspace = config.get("workspace") or defaults.get("workspace", "default")
    domain = config.get("domain") or defaults.get("domain")

    return name, workspace, domain


def _extract_and_validate_explicit_ids(
    config: dict[str, Any],
) -> dict[str, str | None]:
    """Extract and validate explicit ID overrides from config.

    Returns:
        Dict with keys: profile_id, workspace_id, domain_id (values may be None).

    Raises:
        ValidationError: If any explicit ID is invalid.
    """
    explicit_profile_id = config.get("profile_id")
    explicit_workspace_id = config.get("workspace_id")
    explicit_domain_id = config.get("domain_id")

    # Validate explicit IDs if provided
    if explicit_profile_id is not None:
        _validate_id(explicit_profile_id, "profile_id")
    if explicit_workspace_id is not None:
        _validate_id(explicit_workspace_id, "workspace_id")
    if explicit_domain_id is not None:
        _validate_id(explicit_domain_id, "domain_id")

    return {
        "profile_id": explicit_profile_id,
        "workspace_id": explicit_workspace_id,
        "domain_id": explicit_domain_id,
    }


def _resolve_ids(
    name: str,
    workspace: str,
    domain: str | None,
    explicit_ids: dict[str, str | None],
) -> tuple[str | None, str, str]:
    """Resolve final IDs from names and explicit overrides.

    Returns:
        Tuple of (domain_id, workspace_id, profile_id).
    """
    # Validate: cannot provide explicit domain_id without domain name
    if explicit_ids["domain_id"] is not None and domain is None:
        raise ValidationError("Cannot provide explicit domain_id without domain name")

    domain_id: str | None = None
    if domain is not None:
        domain_id = explicit_ids["domain_id"] or Domain.generate_id(domain)

    workspace_id = explicit_ids["workspace_id"] or Workspace.generate_id(domain, workspace)

    profile_id = explicit_ids["profile_id"] or Profile.generate_id(domain, workspace, name)

    return domain_id, workspace_id, profile_id


def _validate_id(id_str: str, name: str) -> None:
    """Validate that an ID is a proper 32-character hex hash.

    Args:
        id_str: The ID string to validate.
        name: Field name for error messages.

    Raises:
        ValidationError: If ID format is invalid.
    """
    if not isinstance(id_str, str):
        raise ValidationError(f"{name} must be a string, got {type(id_str).__name__}")
    if len(id_str) != 32:
        raise ValidationError(f"{name} must be 32 characters, got {len(id_str)}: {id_str!r}")
    try:
        int(id_str, 16)
    except ValueError:
        raise ValidationError(f"{name} must be hex string, got: {id_str!r}") from None
