"""Embedding storage - entity-type agnostic vector storage."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from datetime import datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, Index, Integer, String, UniqueConstraint, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base, utc_now
from .exceptions import ValidationError


@contextmanager
def ensure_session(session: Any | None, session_factory: Callable[[], Any]):
    """
    Context manager that either uses provided session or creates a new one.

    If session is provided, yields it without committing (caller controls transaction).
    If session is None, creates new session and commits on successful exit.

    Args:
        session: Optional existing session to use.
        session_factory: Factory to create new session if needed.

    Yields:
        Database session to use.
    """
    if session is not None:
        # Use provided session, don't commit (caller controls transaction)
        yield session
    else:
        # Create new session and commit on success
        with session_factory() as sess:
            yield sess
            sess.commit()


def _validate_embedding(embedding: list[float] | None) -> None:
    """
    Validate an embedding vector.

    Checks that:
    - Embedding is not None or empty
    - All values are numeric (int or float)
    - All values are finite (not NaN, inf, -inf)

    Args:
        embedding: The embedding vector to validate.

    Raises:
        ValidationError: If validation fails.
    """
    if not embedding:
        raise ValidationError("embedding cannot be empty")

    import math

    for i, val in enumerate(embedding):
        if not isinstance(val, (int, float)):
            raise ValidationError(
                f"embedding values must be numeric, got {type(val).__name__} at index {i}"
            )
        if math.isnan(val) or math.isinf(val):
            raise ValidationError(f"embedding values must be finite, got {val} at index {i}")


class Embedding(Base):
    """
    Vector embedding storage - entity-type agnostic.

    Stores embeddings for any entity type (facts, nodes, content, etc.).
    Each memory model uses a distinct entity_type prefix to namespace its embeddings.

    The entity_id is a string to accommodate both integer IDs (cast to string)
    and hash-based IDs used by profiles, workspaces, and domains.

    Multiple embedding models can be used for the same entity - each model
    gets its own record, allowing model upgrades without data loss.
    """

    __tablename__ = "embeddings"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    entity_type: Mapped[str] = mapped_column(String(50), nullable=False)
    entity_id: Mapped[str] = mapped_column(String(64), nullable=False)
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    dimensions: Mapped[int] = mapped_column(Integer, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )

    __table_args__ = (
        UniqueConstraint(
            "entity_type", "entity_id", "model_name", name="uq_embedding_entity_model"
        ),
        Index("idx_embedding_entity", "entity_type", "entity_id"),
        Index("idx_embedding_model", "model_name"),
    )

    def __repr__(self) -> str:
        return (
            f"<Embedding(id={self.id}, entity_type={self.entity_type!r}, "
            f"entity_id={self.entity_id!r}, model={self.model_name!r})>"
        )


class EmbeddingStore:
    """
    Vector storage and search operations.

    Provides entity-type agnostic operations for storing, searching, and
    managing embeddings. Memory models use this with their own entity_type
    prefix to namespace their embeddings.

    Example:
        store = EmbeddingStore(session_factory)

        # Store embedding for a fact
        store.store("atomic.fact", "123", embedding, "text-embedding-3-small")

        # Search for similar facts
        results = store.search(query_embedding, "atomic.fact", "text-embedding-3-small")

        # Delete when fact is deleted
        store.delete("atomic.fact", "123")
    """

    def __init__(self, session_factory: Callable[[], Any]) -> None:
        """
        Initialize EmbeddingStore.

        Args:
            session_factory: Callable that returns a context manager for database sessions.
        """
        self._session_factory = session_factory

    @staticmethod
    def _update_embedding(record: Embedding, embedding: list[float]) -> Embedding:
        """Update an existing embedding record with new vector data."""
        record.embedding = embedding
        record.dimensions = len(embedding)
        return record

    @staticmethod
    def _create_embedding(
        entity_type: str, entity_id: str, model_name: str, embedding: list[float]
    ) -> Embedding:
        """Create a new Embedding record."""
        return Embedding(
            entity_type=entity_type,
            entity_id=entity_id,
            model_name=model_name,
            dimensions=len(embedding),
            embedding=embedding,
        )

    def store(
        self,
        entity_type: str,
        entity_id: str,
        embedding: list[float],
        model_name: str,
        session: Any | None = None,
    ) -> Embedding:
        """
        Store embedding, replacing existing if present (upsert).

        Args:
            entity_type: Type prefix (e.g., "atomic.fact", "graph.node").
            entity_id: Entity ID (string representation).
            embedding: Vector embedding.
            model_name: Embedding model name.
            session: Optional session to use. If None, creates new session and commits.
                     If provided, uses existing session without committing (caller controls).

        Returns:
            The stored Embedding record.

        Raises:
            ValidationError: If embedding is empty or contains non-finite values.
        """
        _validate_embedding(embedding)

        with ensure_session(session, self._session_factory) as sess:
            # Check for existing
            stmt = select(Embedding).where(
                Embedding.entity_type == entity_type,
                Embedding.entity_id == entity_id,
                Embedding.model_name == model_name,
            )
            existing: Embedding | None = sess.scalar(stmt)
            if existing:
                self._update_embedding(existing, embedding)
                return existing

            emb = self._create_embedding(entity_type, entity_id, model_name, embedding)
            sess.add(emb)
            try:
                sess.flush()
            except IntegrityError:
                sess.rollback()
                existing = sess.scalar(stmt)
                if existing:
                    self._update_embedding(existing, embedding)
                    return existing
                raise
            sess.refresh(emb)
            return emb

    def search(
        self,
        query: list[float],
        entity_type: str,
        model_name: str,
        *,
        top_k: int = 10,
        min_similarity: float = 0.0,
    ) -> list[tuple[str, float]]:
        """
        Search for similar entities by vector similarity.

        Uses cosine distance for similarity scoring. Results are ordered
        by similarity (highest first).

        Args:
            query: Query embedding vector.
            entity_type: Type prefix to search within.
            model_name: Embedding model to search.
            top_k: Maximum results to return.
            min_similarity: Minimum similarity threshold (0.0-1.0).

        Returns:
            List of (entity_id, similarity) tuples, ordered by similarity descending.
        """
        with self._session_factory() as session:
            # Cosine similarity = 1 - cosine_distance
            similarity = (1 - Embedding.embedding.cosine_distance(query)).label("similarity")

            stmt = (
                select(Embedding.entity_id, similarity)
                .where(
                    Embedding.entity_type == entity_type,
                    Embedding.model_name == model_name,
                )
                .order_by(Embedding.embedding.cosine_distance(query))
                .limit(top_k)
            )

            results = session.execute(stmt).all()

            # Filter by minimum similarity
            return [(entity_id, sim) for entity_id, sim in results if sim >= min_similarity]

    def get(
        self,
        entity_type: str,
        entity_id: str,
        model_name: str,
    ) -> list[float] | None:
        """
        Get embedding for an entity.

        Args:
            entity_type: Type prefix.
            entity_id: Entity ID.
            model_name: Embedding model name.

        Returns:
            Embedding vector if found, None otherwise.
        """
        with self._session_factory() as session:
            stmt = select(Embedding.embedding).where(
                Embedding.entity_type == entity_type,
                Embedding.entity_id == entity_id,
                Embedding.model_name == model_name,
            )
            result = session.scalar(stmt)
            return list(result) if result is not None else None

    def delete(self, entity_type: str, entity_id: str) -> int:
        """
        Delete all embeddings for an entity.

        Should be called when the parent entity is deleted to prevent orphans.

        Args:
            entity_type: Type prefix.
            entity_id: Entity ID.

        Returns:
            Number of embeddings deleted.
        """
        with self._session_factory() as session:
            stmt = select(Embedding).where(
                Embedding.entity_type == entity_type,
                Embedding.entity_id == entity_id,
            )
            embeddings = list(session.scalars(stmt).all())
            count = len(embeddings)
            for emb in embeddings:
                session.delete(emb)
            session.commit()
            return count

    def exists(self, entity_type: str, entity_id: str, model_name: str) -> bool:
        """
        Check if embedding exists for an entity.

        Args:
            entity_type: Type prefix.
            entity_id: Entity ID.
            model_name: Embedding model name.

        Returns:
            True if embedding exists.
        """
        from sqlalchemy import func

        with self._session_factory() as session:
            stmt = (
                select(func.count())
                .select_from(Embedding)
                .where(
                    Embedding.entity_type == entity_type,
                    Embedding.entity_id == entity_id,
                    Embedding.model_name == model_name,
                )
            )
            return (session.scalar(stmt) or 0) > 0

    def list_missing(
        self,
        entity_type: str,
        entity_ids: list[str],
        model_name: str,
    ) -> list[str]:
        """
        Find entity IDs that don't have embeddings for the given model.

        Useful for batch embedding generation.

        Args:
            entity_type: Type prefix.
            entity_ids: List of entity IDs to check.
            model_name: Embedding model name.

        Returns:
            List of entity IDs missing embeddings.
        """
        if not entity_ids:
            return []

        with self._session_factory() as session:
            stmt = select(Embedding.entity_id).where(
                Embedding.entity_type == entity_type,
                Embedding.entity_id.in_(entity_ids),
                Embedding.model_name == model_name,
            )
            existing = set(session.scalars(stmt).all())
            return [eid for eid in entity_ids if eid not in existing]

    def count(self, entity_type: str | None = None, model_name: str | None = None) -> int:
        """
        Count embeddings.

        Args:
            entity_type: Optional type filter.
            model_name: Optional model filter.

        Returns:
            Total count of matching embeddings.
        """
        from sqlalchemy import func

        with self._session_factory() as session:
            stmt = select(func.count()).select_from(Embedding)
            if entity_type:
                stmt = stmt.where(Embedding.entity_type == entity_type)
            if model_name:
                stmt = stmt.where(Embedding.model_name == model_name)
            return session.scalar(stmt) or 0
