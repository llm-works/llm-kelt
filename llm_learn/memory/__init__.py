"""Memory module - knowledge storage implementations.

Submodules:
- atomic: Fact-based storage with type-specific detail tables
- isolation: IsolationContext for data partitioning
"""

from . import atomic
from .isolation import IsolationContext

__all__ = ["atomic", "IsolationContext"]
