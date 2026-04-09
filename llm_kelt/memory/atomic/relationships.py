"""Fact relationships client — graph-like edges between atomic facts."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

from appinfra.log import Logger
from sqlalchemy import delete, func, or_, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import joinedload

from llm_kelt.core.errors import ConflictError, ValidationError

from .models import Fact, FactRelationship, RelType


class RelationshipsClient:
    """
    Client for managing edges between atomic facts.

    Supports typed, directed relationships with optional confidence and metadata.
    Symmetric relationship types (contradicts, related_to) are stored with
    normalized ID ordering so (A, B) and (B, A) map to the same row.

    Usage:
        client = RelationshipsClient(lg, session_factory, context_key)
        client.link(fact_a, fact_b, RelType.CONTRADICTS, metadata={"reason": "..."})
        related = client.get_related(fact_a, RelType.CONTRADICTS)
        chain = client.get_chain(fact_a, RelType.DERIVED_FROM)
    """

    def __init__(
        self,
        lg: Logger,
        session_factory: Callable[[], Any],
        context_key: str | None,
    ) -> None:
        self._lg = lg
        self._session_factory = session_factory
        self.context_key = context_key

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------

    def _build_context_filter(self, column: Any) -> Any:
        """Build context filter with glob pattern support."""
        from llm_kelt.memory.isolation import build_context_filter

        return build_context_filter(self.context_key, column)

    @staticmethod
    def _normalize_ids(source_id: int, target_id: int, rel_type: RelType) -> tuple[int, int]:
        """For symmetric types, return (min, max) to ensure canonical ordering."""
        if rel_type.symmetric:
            return (min(source_id, target_id), max(source_id, target_id))
        return (source_id, target_id)

    def _validate_facts_exist(self, session: Any, fact_ids: list[int]) -> None:
        """Verify facts exist, are active, and belong to context."""
        stmt = select(Fact.id).where(Fact.id.in_(fact_ids), Fact.active == True)  # noqa: E712
        context_filter = self._build_context_filter(Fact.context_key)
        if context_filter is not None:
            stmt = stmt.where(context_filter)

        found = set(session.scalars(stmt).all())
        missing = set(fact_ids) - found
        if missing:
            raise ValidationError(f"Facts not found or inactive: {sorted(missing)}")

    def _validate_link_args(self, source_id: int, target_id: int, confidence: float | None) -> None:
        """Validate link() arguments before touching the database."""
        if source_id == target_id:
            raise ValidationError("Cannot create a relationship from a fact to itself")
        if confidence is not None and not (0.0 <= confidence <= 1.0):
            raise ValidationError(f"confidence must be between 0.0 and 1.0, got {confidence}")

    def _build_unlink_stmt(self, source_id: int, target_id: int, rel_type: RelType | None) -> Any:
        """Build the DELETE statement for unlink(), handling symmetric normalization."""
        if rel_type is not None and rel_type.symmetric:
            src, tgt = self._normalize_ids(source_id, target_id, rel_type)
            return delete(FactRelationship).where(
                FactRelationship.source_id == src,
                FactRelationship.target_id == tgt,
                FactRelationship.relationship_type == rel_type.db_value,
            )
        if rel_type is not None:
            return delete(FactRelationship).where(
                FactRelationship.source_id == source_id,
                FactRelationship.target_id == target_id,
                FactRelationship.relationship_type == rel_type.db_value,
            )
        # No type: delete all edges in both directions
        return delete(FactRelationship).where(
            or_(
                (FactRelationship.source_id == source_id)
                & (FactRelationship.target_id == target_id),
                (FactRelationship.source_id == target_id)
                & (FactRelationship.target_id == source_id),
            )
        )

    def _query_relationships(self) -> Any:
        """Base SELECT for relationship queries with eager-loaded facts."""
        return select(FactRelationship).options(
            joinedload(FactRelationship.source_fact),
            joinedload(FactRelationship.target_fact),
        )

    def _build_chain_context_filter(self) -> tuple[str, dict[str, Any]]:
        """Build SQL fragment and params for context filtering in raw CTE queries."""
        if self.context_key is None:
            return ("", {})

        if "*" in self.context_key or "?" in self.context_key:
            pattern = self.context_key.replace("%", r"\%").replace("_", r"\_")
            pattern = pattern.replace("*", "%").replace("?", "_")
            return (
                r"AND r.context_key LIKE :ctx_key ESCAPE '\'",
                {"ctx_key": pattern},
            )

        return ("AND r.context_key = :ctx_key", {"ctx_key": self.context_key})

    def _build_chain_sql(self, ctx_clause: str) -> Any:
        """Build the recursive CTE SQL for get_chain()."""
        return text(f"""
            WITH RECURSIVE chain AS (
                SELECT r.id, r.source_id, r.target_id, 1 AS depth
                FROM atomic_fact_relationships r
                WHERE r.source_id = :fact_id
                  AND r.relationship_type = :rel_type
                  {ctx_clause}

                UNION ALL

                SELECT r.id, r.source_id, r.target_id, c.depth + 1
                FROM atomic_fact_relationships r
                JOIN chain c ON r.source_id = c.target_id
                WHERE r.relationship_type = :rel_type
                  AND c.depth < :max_depth
                  {ctx_clause}
            )
            CYCLE target_id SET is_cycle USING path
            SELECT id FROM chain WHERE NOT is_cycle
            ORDER BY depth
            LIMIT :max_results
        """)

    def _load_relationships_by_ids(
        self, session: Any, rel_ids: list[int]
    ) -> list[FactRelationship]:
        """Load full ORM relationship objects by ID, preserving order."""
        stmt = self._query_relationships().where(FactRelationship.id.in_(rel_ids))
        rels = list(session.scalars(stmt).unique().all())

        id_order = {rid: i for i, rid in enumerate(rel_ids)}
        rels.sort(key=lambda r: id_order.get(r.id, 0))
        return _detach_relationships(rels, session)

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def link(
        self,
        source_id: int,
        target_id: int,
        rel_type: RelType,
        confidence: float | None = 1.0,
        metadata: dict | None = None,
    ) -> int:
        """
        Create a relationship edge between two facts.

        Returns the relationship ID.

        Raises:
            ValidationError: If source == target, confidence out of range,
                or either fact doesn't exist / is inactive.
            ConflictError: If a duplicate relationship already exists.
        """
        self._validate_link_args(source_id, target_id, confidence)
        src, tgt = self._normalize_ids(source_id, target_id, rel_type)

        with self._session_factory() as session:
            self._validate_facts_exist(session, [src, tgt])
            rel = FactRelationship(
                source_id=src,
                target_id=tgt,
                relationship_type=rel_type.db_value,
                confidence=confidence,
                metadata_=metadata,
                context_key=self.context_key,
            )
            session.add(rel)
            try:
                session.flush()
            except IntegrityError as e:
                session.rollback()
                raise ConflictError(
                    f"Relationship already exists: {src}-[{rel_type.db_value}]->{tgt}"
                ) from e

            self._lg.debug(
                "linked facts",
                extra={"source_id": src, "target_id": tgt, "type": rel_type.db_value},
            )
            return rel.id

    def unlink(
        self,
        source_id: int,
        target_id: int,
        rel_type: RelType | None = None,
    ) -> int:
        """
        Remove relationship edge(s) between two facts.

        Returns number of relationships deleted.
        """
        with self._session_factory() as session:
            stmt = self._build_unlink_stmt(source_id, target_id, rel_type)
            context_filter = self._build_context_filter(FactRelationship.context_key)
            if context_filter is not None:
                stmt = stmt.where(context_filter)

            result = session.execute(stmt)
            count: int = result.rowcount
            if count > 0:
                self._lg.debug(
                    "unlinked facts",
                    extra={
                        "source_id": source_id,
                        "target_id": target_id,
                        "type": rel_type.db_value if rel_type else None,
                        "deleted": count,
                    },
                )
            return count

    def get_related(
        self,
        fact_id: int,
        rel_type: RelType | None = None,
        direction: Literal["outgoing", "incoming", "both"] = "both",
    ) -> list[FactRelationship]:
        """
        Get relationships for a fact.

        For symmetric types, direction is always treated as "both".
        Returns FactRelationship objects with source_fact and target_fact loaded.
        """
        effective_dir = "both" if (rel_type and rel_type.symmetric) else direction

        with self._session_factory() as session:
            stmt = self._query_relationships()
            stmt = stmt.where(_direction_filter(fact_id, effective_dir))

            if rel_type is not None:
                stmt = stmt.where(FactRelationship.relationship_type == rel_type.db_value)

            context_filter = self._build_context_filter(FactRelationship.context_key)
            if context_filter is not None:
                stmt = stmt.where(context_filter)

            stmt = stmt.order_by(FactRelationship.created_at.desc())
            rels = list(session.scalars(stmt).unique().all())
            return _detach_relationships(rels, session)

    def find_contradictions(self, fact_id: int | None = None) -> list[FactRelationship]:
        """
        Find contradiction relationships.

        If fact_id provided, find contradictions for that fact.
        If None, find all contradictions in the context.
        """
        if fact_id is not None:
            return self.get_related(fact_id, rel_type=RelType.CONTRADICTS)

        with self._session_factory() as session:
            stmt = self._query_relationships().where(
                FactRelationship.relationship_type == RelType.CONTRADICTS.db_value
            )
            context_filter = self._build_context_filter(FactRelationship.context_key)
            if context_filter is not None:
                stmt = stmt.where(context_filter)

            stmt = stmt.order_by(FactRelationship.created_at.desc()).limit(1000)
            rels = list(session.scalars(stmt).unique().all())
            return _detach_relationships(rels, session)

    def get_chain(
        self,
        fact_id: int,
        rel_type: RelType = RelType.DERIVED_FROM,
        max_depth: int = 5,
        max_results: int = 100,
    ) -> list[FactRelationship]:
        """
        Traverse a chain of relationships using a recursive CTE.

        Follows edges from fact_id outward (source -> target) up to max_depth levels.
        Uses PG14+ CYCLE detection to handle circular references safely.
        Returns a flat list of FactRelationship objects ordered by depth.
        """
        if max_depth < 1:
            raise ValidationError("max_depth must be >= 1")
        if max_results < 1:
            raise ValidationError("max_results must be >= 1")

        ctx_clause, ctx_params = self._build_chain_context_filter()
        sql = self._build_chain_sql(ctx_clause)
        params = {
            "fact_id": fact_id,
            "rel_type": rel_type.db_value,
            "max_depth": max_depth,
            "max_results": max_results,
            **ctx_params,
        }

        with self._session_factory() as session:
            result = session.execute(sql, params)
            rel_ids = [row[0] for row in result]
            if not rel_ids:
                return []
            return self._load_relationships_by_ids(session, rel_ids)

    def count(self, rel_type: RelType | None = None) -> int:
        """Count relationship edges in the context."""
        with self._session_factory() as session:
            stmt = select(func.count()).select_from(FactRelationship)

            if rel_type is not None:
                stmt = stmt.where(FactRelationship.relationship_type == rel_type.db_value)

            context_filter = self._build_context_filter(FactRelationship.context_key)
            if context_filter is not None:
                stmt = stmt.where(context_filter)

            return session.scalar(stmt) or 0


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _direction_filter(fact_id: int, direction: str) -> Any:
    """Build SQLAlchemy WHERE clause for edge direction."""
    if direction == "outgoing":
        return FactRelationship.source_id == fact_id
    if direction == "incoming":
        return FactRelationship.target_id == fact_id
    return or_(
        FactRelationship.source_id == fact_id,
        FactRelationship.target_id == fact_id,
    )


def _detach_relationships(rels: list[FactRelationship], session: Any) -> list[FactRelationship]:
    """Detach relationship objects and their loaded facts from the session.

    Tracks already-detached facts to avoid expunging the same object twice
    (e.g., when a fact appears as target in one edge and source in another).
    """
    from appinfra.db.utils import detach

    detached_facts: set[int] = set()
    for rel in rels:
        for fact in (rel.source_fact, rel.target_fact):
            if fact is not None and fact.id not in detached_facts:
                detach(fact, session)
                detached_facts.add(fact.id)
        detach(rel, session)
    return rels
