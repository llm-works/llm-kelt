"""Base protocol for embedding stores."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Protocol

from sqlalchemy.orm import DeclarativeBase

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from ..types import Calibration


class EmbeddingBase(DeclarativeBase):
    """Separate base for dynamic embedding models.

    Embedding tables are created on-demand, not via migrations. Using a separate
    base prevents them from polluting the main Base.metadata used by schema
    migrations and ensures test isolation.
    """

    pass


class EmbeddingStoreProtocol(Protocol):
    """Protocol for format-specific embedding stores."""

    def store(
        self,
        entity_type: str,
        entity_id: str,
        embedding: list[float],
        model_name: str,
        calibration: Calibration | None = None,
        session: Session | None = None,
    ) -> None:
        """Store an embedding.

        Args:
            entity_type: Type prefix (e.g., "atomic.fact").
            entity_id: Entity identifier.
            embedding: Float32 embedding vector.
            model_name: Embedding model name.
            calibration: Optional calibration data for quantized formats.
            session: Optional session for transaction participation.
        """
        ...

    def search(
        self,
        query: list[float],
        entity_type: str,
        model_name: str,
        top_k: int,
        min_similarity: float = 0.0,
        entity_id_subquery: Any | None = None,
    ) -> list[tuple[str, float]]:
        """Search for similar embeddings.

        Args:
            query: Query embedding vector (float32).
            entity_type: Type prefix to search within.
            model_name: Embedding model to search.
            top_k: Maximum results to return.
            min_similarity: Minimum similarity threshold.
            entity_id_subquery: Optional subquery for pre-filtering.

        Returns:
            List of (entity_id, similarity) tuples, ordered by similarity descending.
        """
        ...

    def get(
        self,
        entity_type: str,
        entity_id: str,
        model_name: str,
    ) -> list[float] | None:
        """Get embedding for an entity (dequantized to float32).

        Args:
            entity_type: Type prefix.
            entity_id: Entity identifier.
            model_name: Embedding model name.

        Returns:
            Float32 embedding vector if found, None otherwise.
        """
        ...

    def delete(
        self,
        entity_type: str,
        entity_id: str,
        session: Session | None = None,
    ) -> int:
        """Delete embeddings for an entity.

        Args:
            entity_type: Type prefix.
            entity_id: Entity identifier.
            session: Optional session for transaction participation.

        Returns:
            Number of embeddings deleted.
        """
        ...

    def exists(
        self,
        entity_type: str,
        entity_id: str,
        model_name: str,
    ) -> bool:
        """Check if embedding exists for an entity.

        Args:
            entity_type: Type prefix.
            entity_id: Entity identifier.
            model_name: Embedding model name.

        Returns:
            True if embedding exists.
        """
        ...

    def count(
        self,
        entity_type: str | None = None,
        model_name: str | None = None,
    ) -> int:
        """Count embeddings.

        Args:
            entity_type: Optional type filter.
            model_name: Optional model filter.

        Returns:
            Total count of matching embeddings.
        """
        ...

    def ensure_table(self) -> None:
        """Ensure the embedding table exists (on-demand creation).

        Creates the table if it doesn't exist, including any format-specific
        indexes (e.g., HNSW for pgvector types).
        """
        ...


@contextmanager
def ensure_session(session: Any | None, session_factory: Callable[[], Any]):
    """Context manager that uses provided session or creates new one.

    If session is provided, yields it without committing.
    If session is None, creates new session via factory.
    """
    if session is not None:
        yield session
    else:
        with session_factory() as sess:
            yield sess


def table_exists(conn: Any, table_name: str) -> bool:
    """Check if a table exists in the current search_path."""
    from sqlalchemy import text

    result = conn.execute(
        text(
            "SELECT 1 FROM pg_catalog.pg_class c "
            "JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace "
            "WHERE c.relname = :table_name "
            "AND pg_catalog.pg_table_is_visible(c.oid)"
        ).bindparams(table_name=table_name)
    )
    return result.scalar() is not None


def index_exists(conn: Any, index_name: str) -> bool:
    """Check if an index exists in the database."""
    from sqlalchemy import text

    result = conn.execute(
        text("SELECT 1 FROM pg_indexes WHERE indexname = :idx_name").bindparams(idx_name=index_name)
    )
    return result.scalar() is not None
