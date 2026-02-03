"""Memory module - knowledge storage implementations.

Each submodule implements a different knowledge representation model:
- atomic: Fact-based storage with type-specific detail tables
"""

from . import atomic

__all__ = ["atomic"]
