"""Memory module - knowledge storage implementations.

Submodules:
- atomic: Fact-based storage with type-specific detail tables
- isolation: ClientContext for data partitioning
"""

from . import atomic
from .isolation import ClientContext

__all__ = ["atomic", "ClientContext"]
