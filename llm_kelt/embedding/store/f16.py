"""Float16 embedding store using pgvector halfvec type."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

from sqlalchemy import delete, func, select, text
from sqlalchemy.exc import IntegrityError

from .base import ensure_session, index_exists, table_exists

if TYPE_CHECKING:
    from ..types import Calibration


class Float16Store:
    """Float16 embedding store using pgvector halfvec type.

    Uses native pgvector halfvec for 2x storage reduction with ~0% accuracy loss.
    Embeddings are converted to float16 on storage, returned as float32 on retrieval.
    """

    def __init__(self, session_factory: Callable[[], Any], dimensions: int, model: Any) -> None:
        """Initialize Float16Store.

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
        """Create table and HNSW index if they don't exist."""
        if self._table_ensured:
            return

        with self._session_factory() as session:
            conn = session.connection()

            if table_exists(conn, self.table_name):
                self._table_ensured = True
                return

            self._model.__table__.create(conn, checkfirst=True)

            hnsw_idx = f"idx_{self.table_name}_hnsw"
            if not index_exists(conn, hnsw_idx):
                conn.execute(
                    text(
                        f"CREATE INDEX {hnsw_idx} ON {self.table_name} "
                        f"USING hnsw (embedding halfvec_cosine_ops) "
                        f"WITH (m = 16, ef_construction = 64)"
                    )
                )

            session.commit()
        self._table_ensured = True

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
            return result.to_list() if result is not None else None

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
