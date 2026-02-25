"""Shared types for the Kelt framework."""

from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass
class ScoredEntity(Generic[T]):
    """
    Entity with similarity score from vector search.

    Used to return search results with their relevance scores.
    """

    entity: T
    score: float

    def __repr__(self) -> str:
        return f"<ScoredEntity(score={self.score:.4f}, entity={self.entity!r})>"


@dataclass
class PagedResult(Generic[T]):
    """
    Paginated result set.

    Contains a page of items along with pagination metadata.
    """

    items: list[T]
    total: int
    offset: int
    limit: int

    @property
    def has_more(self) -> bool:
        """Check if there are more items beyond this page."""
        return self.offset + len(self.items) < self.total

    @property
    def page_count(self) -> int:
        """Total number of pages."""
        if self.limit <= 0:
            return 0
        return (self.total + self.limit - 1) // self.limit

    @property
    def current_page(self) -> int:
        """Current page number (1-indexed)."""
        if self.limit <= 0:
            return 1
        return (self.offset // self.limit) + 1

    def __repr__(self) -> str:
        return (
            f"<PagedResult(items={len(self.items)}, total={self.total}, "
            f"offset={self.offset}, limit={self.limit})>"
        )
