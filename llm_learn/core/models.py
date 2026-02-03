"""SQLAlchemy ORM models for Learn framework.

DEPRECATED: Import from individual modules instead:
    - from llm_learn.core.base import Base
    - from llm_learn.core.domain import Domain
    - from llm_learn.core.workspace import Workspace
    - from llm_learn.core.profile import Profile
    - from llm_learn.core.content import Content

This module re-exports for backwards compatibility.
"""

# Re-export from new modules for backwards compatibility
from .base import Base
from .content import Content
from .domain import Domain
from .profile import Profile
from .workspace import Workspace

__all__ = [
    "Base",
    "Domain",
    "Workspace",
    "Profile",
    "Content",
]
