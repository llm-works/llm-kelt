"""Int4 embedding store using application-level scalar quantization."""

from __future__ import annotations

import heapq
import math
from collections.abc import Callable
from typing import Any

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from ..quantize import dequantize_int4, quantize_int4
from ..types import Calibration, QuantizationFormat, QuantizedEmbedding
from .base import StoreBase, ensure_session, table_exists


class Int4Store(StoreBase):
    """Int4 embedding store using application-level scalar quantization.

    Stores embeddings as packed 4-bit values (2 values per byte) with per-embedding scale/offset.
    Provides 8x storage reduction with ~95-97% accuracy retention.
    Recommended to use with reranking for better accuracy.
    """

    def __init__(self, session_factory: Callable[[], Any], dimensions: int, model: Any) -> None:
        """Initialize Int4Store.

        Args:
            session_factory: Callable that returns a context manager for DB sessions.
            dimensions: Vector dimensions.
            model: SQLAlchemy model class for this store's table.
        """
        self._session_factory = session_factory
        self._dimensions = dimensions
        self._model = model
        self._table_ensured = False

    def ensure_table(self) -> None:
        """Create table if it doesn't exist."""
        if self._table_ensured:
            return

        with self._session_factory() as session:
            conn = session.connection()
            if table_exists(conn, self.table_name):
                self._table_ensured = True
                return
            self._model.__table__.create(conn, checkfirst=True)
            session.commit()
        self._table_ensured = True

    def _update_record(self, record: Any, qemb: QuantizedEmbedding) -> None:
        """Update existing record with quantized data."""
        record.embedding_bytes, record.scale, record.offset = qemb.data, qemb.scale, qemb.offset

    def _upsert(
        self, sess: Any, entity_type: str, entity_id: str, model: str, qemb: QuantizedEmbedding
    ) -> None:
        """Insert or update quantized embedding with race condition handling."""
        stmt = select(self._model).where(
            self._model.entity_type == entity_type,
            self._model.entity_id == entity_id,
            self._model.model_name == model,
        )
        if existing := sess.scalar(stmt):
            self._update_record(existing, qemb)
            return
        record = self._model(
            entity_type=entity_type,
            entity_id=entity_id,
            model_name=model,
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
        model: str,
        calibration: Calibration | None = None,
        session: Any | None = None,
    ) -> None:
        """Store an embedding with scalar quantization (upsert)."""
        if len(embedding) != self._dimensions:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self._dimensions}, got {len(embedding)}"
            )
        quantized = quantize_int4(embedding, calibration)
        with ensure_session(session, self._session_factory) as sess:
            self._upsert(sess, entity_type, entity_id, model, quantized)

    def _row_to_quantized(self, row: Any) -> QuantizedEmbedding:
        """Convert database row to QuantizedEmbedding."""
        return QuantizedEmbedding(
            data=row.embedding_bytes,
            format=QuantizationFormat.I4,
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
        model: str,
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
                self._model.model_name == model,
            )

            if entity_id_subquery is not None:
                stmt = stmt.where(self._model.entity_id.in_(entity_id_subquery))

            heap: list[tuple[float, str]] = []
            for row in session.scalars(stmt):
                qemb = self._row_to_quantized(row)
                dequantized = dequantize_int4(qemb)
                sim = self._cosine_similarity(query, dequantized)
                if sim >= min_similarity:
                    if len(heap) < top_k:
                        heapq.heappush(heap, (sim, row.entity_id))
                    elif sim > heap[0][0]:
                        heapq.heapreplace(heap, (sim, row.entity_id))

            return [(eid, sim) for sim, eid in sorted(heap, reverse=True)]

    def get(
        self,
        entity_type: str,
        entity_id: str,
        model: str,
    ) -> list[float] | None:
        """Get embedding for an entity (dequantized to float32)."""
        with self._session_factory() as session:
            stmt = select(self._model).where(
                self._model.entity_type == entity_type,
                self._model.entity_id == entity_id,
                self._model.model_name == model,
            )
            row = session.scalar(stmt)
            if row is None:
                return None

            qemb = self._row_to_quantized(row)
            return dequantize_int4(qemb)

    def _embedding_from_row(self, row: Any) -> list[float]:
        """Dequantize a row's int4 embedding to float32."""
        return dequantize_int4(self._row_to_quantized(row))

    def get_quantized(
        self,
        entity_type: str,
        entity_id: str,
        model: str,
    ) -> QuantizedEmbedding | None:
        """Get raw quantized embedding without dequantization."""
        with self._session_factory() as session:
            stmt = select(self._model).where(
                self._model.entity_type == entity_type,
                self._model.entity_id == entity_id,
                self._model.model_name == model,
            )
            row = session.scalar(stmt)
            if row is None:
                return None
            return self._row_to_quantized(row)
