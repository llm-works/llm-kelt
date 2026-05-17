"""Float32 embedding store using pgvector vector type."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    DateTime,
    Index,
    String,
    UniqueConstraint,
    delete,
    func,
    select,
)
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Mapped, mapped_column

from llm_kelt.core.base import Base

from .base import ensure_session

if TYPE_CHECKING:
    from ..types import Calibration


def make_f32_model(dimensions: int) -> Any:
    """Create a Float32 embedding model class for specific dimensions.

    Args:
        dimensions: Vector dimensions (e.g., 384, 1536).

    Returns:
        SQLAlchemy model class for the embeddings_{dimensions}_f32 table.
    """
    table_name = f"embeddings_{dimensions}_f32"

    class EmbeddingF32(Base):
        __tablename__ = table_name
        __table_args__ = (
            UniqueConstraint(
                "entity_type", "entity_id", "model_name", name=f"uq_{table_name}_entity_model"
            ),
            Index(f"idx_{table_name}_entity", "entity_type", "entity_id"),
            Index(f"idx_{table_name}_model", "model_name"),
            {"extend_existing": True},
        )

        id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
        entity_type: Mapped[str] = mapped_column(String(50), nullable=False)
        entity_id: Mapped[str] = mapped_column(String(64), nullable=False)
        model_name: Mapped[str] = mapped_column(String(100), nullable=False)
        embedding: Mapped[list[float]] = mapped_column(Vector(dimensions), nullable=False)
        created_at: Mapped[datetime] = mapped_column(
            DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
        )

    EmbeddingF32.__name__ = f"EmbeddingF32_{dimensions}"
    EmbeddingF32.__qualname__ = f"EmbeddingF32_{dimensions}"
    return EmbeddingF32


class Float32Store:
    """Float32 embedding store using pgvector vector type.

    Uses native pgvector for storage and similarity search.
    Full precision, no quantization loss.
    """

    def __init__(self, session_factory: Callable[[], Any], dimensions: int) -> None:
        """Initialize Float32Store."""
        self._session_factory = session_factory
        self._dimensions = dimensions
        self._model = make_f32_model(dimensions)

    @property
    def dimensions(self) -> int:
        """Vector dimensions for this store."""
        return self._dimensions

    @property
    def table_name(self) -> str:
        """Database table name."""
        return cast(str, self._model.__tablename__)

    def _upsert(
        self, sess: Any, entity_type: str, entity_id: str, model_name: str, embedding: list[float]
    ) -> None:
        """Insert or update embedding with race condition handling."""
        stmt = select(self._model).where(
            self._model.entity_type == entity_type,
            self._model.entity_id == entity_id,
            self._model.model_name == model_name,
        )
        existing = sess.scalar(stmt)
        if existing:
            existing.embedding = embedding
            return

        record = self._model(
            entity_type=entity_type, entity_id=entity_id, model_name=model_name, embedding=embedding
        )
        try:
            with sess.begin_nested():
                sess.add(record)
                sess.flush()
        except IntegrityError:
            sess.expunge(record)
            if existing := sess.scalar(stmt):
                existing.embedding = embedding

    def store(
        self,
        entity_type: str,
        entity_id: str,
        embedding: list[float],
        model_name: str,
        calibration: Calibration | None = None,
        session: Any | None = None,
    ) -> None:
        """Store an embedding (upsert)."""
        if len(embedding) != self._dimensions:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self._dimensions}, got {len(embedding)}"
            )
        with ensure_session(session, self._session_factory) as sess:
            self._upsert(sess, entity_type, entity_id, model_name, embedding)

    def search(
        self,
        query: list[float],
        entity_type: str,
        model_name: str,
        top_k: int,
        min_similarity: float = 0.0,
        entity_id_subquery: Any | None = None,
    ) -> list[tuple[str, float]]:
        """Search for similar embeddings using cosine similarity."""
        with self._session_factory() as session:
            similarity = (1 - self._model.embedding.cosine_distance(query)).label("similarity")

            stmt = (
                select(self._model.entity_id, similarity)
                .where(
                    self._model.entity_type == entity_type,
                    self._model.model_name == model_name,
                )
                .order_by(self._model.embedding.cosine_distance(query))
                .limit(top_k)
            )

            if entity_id_subquery is not None:
                stmt = stmt.where(self._model.entity_id.in_(entity_id_subquery))

            results = session.execute(stmt).all()
            return [(eid, sim) for eid, sim in results if sim >= min_similarity]

    def get(
        self,
        entity_type: str,
        entity_id: str,
        model_name: str,
    ) -> list[float] | None:
        """Get embedding for an entity."""
        with self._session_factory() as session:
            stmt = select(self._model.embedding).where(
                self._model.entity_type == entity_type,
                self._model.entity_id == entity_id,
                self._model.model_name == model_name,
            )
            result = session.scalar(stmt)
            return list(result) if result is not None else None

    def delete(
        self,
        entity_type: str,
        entity_id: str,
        session: Any | None = None,
    ) -> int:
        """Delete embeddings for an entity."""

        def _do_delete(sess: Any) -> int:
            stmt = delete(self._model).where(
                self._model.entity_type == entity_type,
                self._model.entity_id == entity_id,
            )
            result = sess.execute(stmt)
            return cast(int, result.rowcount)

        if session is not None:
            return _do_delete(session)

        with self._session_factory() as sess:
            count = _do_delete(sess)
            sess.commit()
            return count

    def exists(
        self,
        entity_type: str,
        entity_id: str,
        model_name: str,
    ) -> bool:
        """Check if embedding exists."""
        with self._session_factory() as session:
            stmt = (
                select(func.count())
                .select_from(self._model)
                .where(
                    self._model.entity_type == entity_type,
                    self._model.entity_id == entity_id,
                    self._model.model_name == model_name,
                )
            )
            return (session.scalar(stmt) or 0) > 0

    def count(
        self,
        entity_type: str | None = None,
        model_name: str | None = None,
    ) -> int:
        """Count embeddings with optional filters."""
        with self._session_factory() as session:
            stmt = select(func.count()).select_from(self._model)
            if entity_type:
                stmt = stmt.where(self._model.entity_type == entity_type)
            if model_name:
                stmt = stmt.where(self._model.model_name == model_name)
            return session.scalar(stmt) or 0
