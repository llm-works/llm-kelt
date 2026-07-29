"""Float32 embedding store using pgvector vector type."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from appinfra.db.pg import index_exists, table_exists, with_object_lock
from sqlalchemy import select, text
from sqlalchemy.exc import IntegrityError

from .base import StoreBase, ensure_session

if TYPE_CHECKING:
    from ..types import Calibration


class Float32Store(StoreBase):
    """Float32 embedding store using pgvector vector type.

    Uses native pgvector for storage and similarity search.
    Full precision, no quantization loss.
    """

    def __init__(self, session_factory: Callable[[], Any], dimensions: int, model: Any) -> None:
        """Initialize Float32Store.

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
        """Create table and HNSW index if they don't exist.

        Concurrent first-touch is serialized by a Postgres advisory lock
        keyed on the table name — see ``appinfra.db.pg.with_object_lock``.
        Steady-state cost is zero: ``_table_ensured`` short-circuits before
        the lock is ever taken.
        """
        if self._table_ensured:
            return

        schema = getattr(self._model.__table__, "schema", None)
        qualified_table = f"{schema}.{self.table_name}" if schema else self.table_name
        hnsw_idx = f"idx_{self.table_name}_hnsw"
        hnsw_sql = text(
            f"CREATE INDEX {hnsw_idx} ON {qualified_table} "
            f"USING hnsw (embedding vector_cosine_ops) "
            f"WITH (m = 16, ef_construction = 64)"
        )

        with self._session_factory() as session:
            conn = session.connection()
            with with_object_lock(session, f"kelt.ensure:{schema}.{self.table_name}"):
                if not table_exists(conn, self.table_name, schema=schema):
                    self._model.__table__.create(conn)
                if not index_exists(conn, hnsw_idx, schema=schema):
                    conn.execute(hnsw_sql)
            session.commit()
        self._table_ensured = True

    def _upsert(
        self, sess: Any, entity_type: str, entity_id: str, model: str, embedding: list[float]
    ) -> None:
        """Insert or update embedding with race condition handling."""
        stmt = select(self._model).where(
            self._model.entity_type == entity_type,
            self._model.entity_id == entity_id,
            self._model.model_name == model,
        )
        existing = sess.scalar(stmt)
        if existing:
            existing.embedding = embedding
            return

        record = self._model(
            entity_type=entity_type, entity_id=entity_id, model_name=model, embedding=embedding
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
        model: str,
        calibration: Calibration | None = None,
        session: Any | None = None,
    ) -> None:
        """Store an embedding (upsert)."""
        if len(embedding) != self._dimensions:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self._dimensions}, got {len(embedding)}"
            )
        with ensure_session(session, self._session_factory) as sess:
            self._upsert(sess, entity_type, entity_id, model, embedding)

    def search(
        self,
        query: list[float],
        entity_type: str,
        model: str,
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
                    self._model.model_name == model,
                    self._model.embedding.cosine_distance(query) <= (1 - min_similarity),
                )
                .order_by(self._model.embedding.cosine_distance(query))
                .limit(top_k)
            )

            if entity_id_subquery is not None:
                stmt = stmt.where(self._model.entity_id.in_(entity_id_subquery))

            results = session.execute(stmt).all()
            return [(eid, float(sim)) for eid, sim in results]

    def get(
        self,
        entity_type: str,
        entity_id: str,
        model: str,
    ) -> list[float] | None:
        """Get embedding for an entity."""
        with self._session_factory() as session:
            stmt = select(self._model.embedding).where(
                self._model.entity_type == entity_type,
                self._model.entity_id == entity_id,
                self._model.model_name == model,
            )
            result = session.scalar(stmt)
            return list(result) if result is not None else None

    def _embedding_from_row(self, row: Any) -> list[float]:
        """Decode a row's vector embedding to float32."""
        return list(row.embedding)
