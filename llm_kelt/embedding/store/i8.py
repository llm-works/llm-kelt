"""Int8 embedding store using application-level scalar quantization."""

from __future__ import annotations

import math
from collections.abc import Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast

from sqlalchemy import (
    BigInteger,
    DateTime,
    Float,
    Index,
    LargeBinary,
    String,
    UniqueConstraint,
    delete,
    func,
    select,
)
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Mapped, mapped_column

from llm_kelt.core.base import Base

from ..quantize import dequantize_int8, quantize_int8
from ..types import Calibration, QuantizationFormat, QuantizedEmbedding
from .base import ensure_session

if TYPE_CHECKING:
    pass


def make_i8_model(dimensions: int) -> Any:
    """Create an Int8 embedding model class for specific dimensions.

    Args:
        dimensions: Vector dimensions (e.g., 384, 1536).

    Returns:
        SQLAlchemy model class for the embeddings_{dimensions}_i8 table.
    """
    table_name = f"embeddings_{dimensions}_i8"

    class EmbeddingI8(Base):
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
        embedding_bytes: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
        scale: Mapped[float] = mapped_column(Float, nullable=False)
        offset: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
        created_at: Mapped[datetime] = mapped_column(
            DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
        )

    EmbeddingI8.__name__ = f"EmbeddingI8_{dimensions}"
    EmbeddingI8.__qualname__ = f"EmbeddingI8_{dimensions}"
    return EmbeddingI8


class Int8Store:
    """Int8 embedding store using application-level scalar quantization.

    Stores embeddings as quantized uint8 bytes with per-embedding scale/offset.
    Provides 4x storage reduction with ~99% accuracy retention.
    Search requires dequantization or approximate int8 distance computation.
    """

    def __init__(self, session_factory: Callable[[], Any], dimensions: int) -> None:
        """Initialize Int8Store."""
        self._session_factory = session_factory
        self._dimensions = dimensions
        self._model = make_i8_model(dimensions)

    @property
    def dimensions(self) -> int:
        """Vector dimensions for this store."""
        return self._dimensions

    @property
    def table_name(self) -> str:
        """Database table name."""
        return cast(str, self._model.__tablename__)

    def _update_record(self, record: Any, qemb: QuantizedEmbedding) -> None:
        """Update existing record with quantized data."""
        record.embedding_bytes, record.scale, record.offset = qemb.data, qemb.scale, qemb.offset

    def _upsert(
        self, sess: Any, entity_type: str, entity_id: str, model_name: str, qemb: QuantizedEmbedding
    ) -> None:
        """Insert or update quantized embedding with race condition handling."""
        stmt = select(self._model).where(
            self._model.entity_type == entity_type,
            self._model.entity_id == entity_id,
            self._model.model_name == model_name,
        )
        if existing := sess.scalar(stmt):
            self._update_record(existing, qemb)
            return
        record = self._model(
            entity_type=entity_type,
            entity_id=entity_id,
            model_name=model_name,
            embedding_bytes=qemb.data,
            scale=qemb.scale,
            offset=qemb.offset,
        )
        try:
            with sess.begin_nested():
                sess.add(record)
                sess.flush()
        except IntegrityError:
            sess.expunge(record)
            if existing := sess.scalar(stmt):
                self._update_record(existing, qemb)

    def store(
        self,
        entity_type: str,
        entity_id: str,
        embedding: list[float],
        model_name: str,
        calibration: Calibration | None = None,
        session: Any | None = None,
    ) -> None:
        """Store an embedding with scalar quantization (upsert)."""
        if len(embedding) != self._dimensions:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self._dimensions}, got {len(embedding)}"
            )
        quantized = quantize_int8(embedding, calibration)
        with ensure_session(session, self._session_factory) as sess:
            self._upsert(sess, entity_type, entity_id, model_name, quantized)

    def _row_to_quantized(self, row: Any) -> QuantizedEmbedding:
        """Convert database row to QuantizedEmbedding."""
        return QuantizedEmbedding(
            data=row.embedding_bytes,
            format=QuantizationFormat.I8,
            dimensions=self._dimensions,
            scale=row.scale,
            offset=row.offset,
        )

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def search(
        self,
        query: list[float],
        entity_type: str,
        model_name: str,
        top_k: int,
        min_similarity: float = 0.0,
        entity_id_subquery: Any | None = None,
    ) -> list[tuple[str, float]]:
        """Search using application-level dequantization and scoring.

        This fetches all matching embeddings, dequantizes them, and computes
        cosine similarity in Python. Suitable for <100k embeddings per query scope.
        """
        with self._session_factory() as session:
            stmt = select(self._model).where(
                self._model.entity_type == entity_type,
                self._model.model_name == model_name,
            )

            if entity_id_subquery is not None:
                stmt = stmt.where(self._model.entity_id.in_(entity_id_subquery))

            rows = list(session.scalars(stmt).all())

            results = []
            for row in rows:
                qemb = self._row_to_quantized(row)
                dequantized = dequantize_int8(qemb)
                sim = self._cosine_similarity(query, dequantized)
                if sim >= min_similarity:
                    results.append((row.entity_id, sim))

            results.sort(key=lambda x: x[1], reverse=True)
            return results[:top_k]

    def get(
        self,
        entity_type: str,
        entity_id: str,
        model_name: str,
    ) -> list[float] | None:
        """Get embedding for an entity (dequantized to float32)."""
        with self._session_factory() as session:
            stmt = select(self._model).where(
                self._model.entity_type == entity_type,
                self._model.entity_id == entity_id,
                self._model.model_name == model_name,
            )
            row = session.scalar(stmt)
            if row is None:
                return None

            qemb = self._row_to_quantized(row)
            return dequantize_int8(qemb)

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

    def get_quantized(
        self,
        entity_type: str,
        entity_id: str,
        model_name: str,
    ) -> QuantizedEmbedding | None:
        """Get raw quantized embedding without dequantization."""
        with self._session_factory() as session:
            stmt = select(self._model).where(
                self._model.entity_type == entity_type,
                self._model.entity_id == entity_id,
                self._model.model_name == model_name,
            )
            row = session.scalar(stmt)
            if row is None:
                return None
            return self._row_to_quantized(row)
