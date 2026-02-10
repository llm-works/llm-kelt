"""SQLAlchemy ORM models for Learn framework.

DEPRECATED: Import from individual modules instead:
    - from llm_learn.core.base import Base
    - from llm_learn.core.content import Content

This module re-exports for backwards compatibility.
"""

# Re-export from new modules for backwards compatibility
from .base import Base
from .content import Content

__all__ = [
    "Base",
    "Content",
]
