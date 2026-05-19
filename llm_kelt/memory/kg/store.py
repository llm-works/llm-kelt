"""Knowledge Graph store - CRUD and query operations with scoped subgraphs."""

from __future__ import annotations

from collections.abc import Callable, Generator
from contextlib import AbstractContextManager, contextmanager
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast

from appinfra.db.utils import detach, detach_all
from sqlalchemy import func, or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session as SASession

from llm_kelt.embedding import Factory as EmbeddingFactory
from llm_kelt.embedding.types import QuantizationFormat

from .models import Entity, EntityAlias, EntityRef, EntityRelationship, FactEntity

if TYPE_CHECKING:
    from appinfra.log import Logger
    from llm_infer.client import EmbeddingClient

    from .embedding import EntityEmbeddingAdapter

SessionFactory = Callable[[], AbstractContextManager[SASession]]

__all__ = [
    "AliasConflictError",
    "KGStore",
    "EntityStore",
    "EntityRefStore",
    "EntityRelationshipStore",
    "FactEntityStore",
]


class AliasConflictError(Exception):
    """Raised when an alias already belongs to a different entity."""

    def __init__(self, entity_id: int, alias: str) -> None:
        self.entity_id = entity_id
        self.alias = alias
        super().__init__(f"Alias '{alias}' belongs to entity {entity_id}")


def scope_ancestors(scope_key: str) -> list[str]:
    """Get all ancestor scopes including the scope itself and global.

    Example: "org:acme:user:alice" -> ["org:acme:user:alice", "org:acme", "global"]
    """
    if scope_key == "global":
        return ["global"]

    ancestors = [scope_key]
    parts = scope_key.split(":")

    while len(parts) > 1:
        parts = parts[:-1]
        # Handle odd-length splits (key:value pairs)
        if len(parts) % 2 == 0:
            ancestors.append(":".join(parts))

    if "global" not in ancestors:
        ancestors.append("global")

    return ancestors


def build_scope_filter(scope_key: str, column: Any) -> Any:
    """Build SQLAlchemy filter for hierarchical scope resolution.

    Returns filter matching scope_key and all ancestor scopes up to global.
    """
    ancestors = scope_ancestors(scope_key)
    return column.in_(ancestors)


class EntityStore:
    """Entity CRUD, dedup, and alias management with scoped subgraphs."""

    def __init__(self, lg: Logger, session_factory: SessionFactory) -> None:
        self._lg = lg
        self._session_factory = session_factory

    def get(self, entity_id: int, *, sa_session: SASession | None = None) -> Entity | None:
        """Get entity by ID."""
        with self._scope(sa_session) as s:
            entity = s.get(Entity, entity_id)
            return detach(entity, s) if entity else None

    def get_by_name(
        self,
        scope_key: str,
        name: str,
        entity_type: str,
        *,
        sa_session: SASession | None = None,
    ) -> Entity | None:
        """Get entity by canonical name within scope hierarchy (case-insensitive)."""
        with self._scope(sa_session) as s:
            stmt = (
                select(Entity)
                .where(
                    build_scope_filter(scope_key, Entity.scope_key),
                    func.lower(Entity.canonical_name) == name.lower().strip(),
                    Entity.entity_type == entity_type,
                )
                .order_by(Entity.id)
            )
            entity = s.scalar(stmt)
            return detach(entity, s) if entity else None

    def get_by_names(
        self,
        scope_key: str,
        names: list[str],
        entity_type: str,
        *,
        sa_session: SASession | None = None,
    ) -> list[Entity]:
        """Get entities by canonical names within scope hierarchy (case-insensitive)."""
        if not names:
            return []

        normalized = [n.lower().strip() for n in names]
        with self._scope(sa_session) as s:
            stmt = select(Entity).where(
                build_scope_filter(scope_key, Entity.scope_key),
                func.lower(Entity.canonical_name).in_(normalized),
                Entity.entity_type == entity_type,
            )
            entities = list(s.scalars(stmt))
            detach_all(entities, s)
            return entities

    def create(
        self,
        scope_key: str,
        canonical_name: str,
        entity_type: str,
        *,
        description: str | None = None,
        extra: dict[str, Any] | None = None,
        aliases: list[str] | None = None,
        sa_session: SASession | None = None,
    ) -> Entity:
        """Create a new entity with optional aliases."""
        with self._scope(sa_session) as s:
            entity = Entity(
                scope_key=scope_key,
                canonical_name=canonical_name.lower().strip(),
                entity_type=entity_type,
                description=description,
                extra=extra or {},
            )
            s.add(entity)
            s.flush()

            canonical_alias = self._add_alias(s, entity, canonical_name, scope_key)
            if canonical_alias.entity_id != entity.id:
                raise AliasConflictError(canonical_alias.entity_id, canonical_name)

            for alias in aliases or []:
                self._add_alias(s, entity, alias, scope_key)

            s.flush()
            s.refresh(entity, ["aliases"])
            return cast(Entity, detach(entity, s))

    def update(
        self,
        entity_id: int,
        *,
        description: str | None = None,
        extra: dict[str, Any] | None = None,
        sa_session: SASession | None = None,
    ) -> Entity | None:
        """Update entity fields."""
        with self._scope(sa_session) as s:
            entity = s.get(Entity, entity_id)
            if entity is None:
                return None
            if description is not None:
                entity.description = description
            if extra is not None:
                entity.extra = {**entity.extra, **extra}
            entity.updated_at = datetime.now(UTC)
            return cast(Entity, detach(entity, s))

    def delete(self, entity_id: int, *, sa_session: SASession | None = None) -> bool:
        """Delete entity and all related data (cascades)."""
        with self._scope(sa_session) as s:
            entity = s.get(Entity, entity_id)
            if entity is None:
                return False
            s.delete(entity)
            return True

    def resolve(
        self,
        scope_key: str,
        name: str,
        entity_type: str | None = None,
        *,
        sa_session: SASession | None = None,
    ) -> Entity | None:
        """Resolve a name to canonical entity via alias lookup within scope hierarchy."""
        normalized = name.lower().strip()
        with self._scope(sa_session) as s:
            stmt = (
                select(Entity)
                .join(EntityAlias, Entity.id == EntityAlias.entity_id)
                .where(
                    build_scope_filter(scope_key, EntityAlias.scope_key),
                    build_scope_filter(scope_key, Entity.scope_key),
                    EntityAlias.alias_normalized == normalized,
                )
                .order_by(Entity.id)
            )
            if entity_type:
                stmt = stmt.where(Entity.entity_type == entity_type)
            entity = s.scalar(stmt)
            return detach(entity, s) if entity else None

    def find_or_create(
        self,
        scope_key: str,
        name: str,
        entity_type: str,
        *,
        description: str | None = None,
        extra: dict[str, Any] | None = None,
        sa_session: SASession | None = None,
    ) -> tuple[int, bool]:
        """Find existing entity or create new one. Returns (entity_id, created).

        Resolution checks scope hierarchy (scope + ancestors up to global).
        Creation happens in the specified scope.
        """
        with self._scope(sa_session) as s:
            entity = self.resolve(scope_key, name, entity_type, sa_session=s)
            if entity:
                return entity.id, False

            try:
                with s.begin_nested():
                    entity = self.create(
                        scope_key,
                        name,
                        entity_type,
                        description=description,
                        extra=extra,
                        sa_session=s,
                    )
                    s.flush()
                return entity.id, True
            except IntegrityError:
                entity = self.resolve(scope_key, name, entity_type, sa_session=s)
                if entity:
                    return entity.id, False
                raise
            except AliasConflictError as e:
                return e.entity_id, False

    def add_alias(
        self,
        entity_id: int,
        alias: str,
        scope_key: str,
        *,
        sa_session: SASession | None = None,
    ) -> EntityAlias | None:
        """Add an alias to an entity. Returns None if alias belongs to another entity."""
        with self._scope(sa_session) as s:
            entity = s.get(Entity, entity_id)
            if entity is None:
                return None
            ea = self._add_alias(s, entity, alias, scope_key)
            if ea.entity_id != entity_id:
                return None
            s.flush()
            return cast(EntityAlias, detach(ea, s))

    def _add_alias(
        self,
        s: SASession,
        entity: Entity,
        alias: str,
        scope_key: str,
    ) -> EntityAlias:
        """Internal: add alias within existing session."""
        normalized = alias.lower().strip()
        existing = s.scalar(
            select(EntityAlias).where(
                EntityAlias.scope_key == scope_key,
                EntityAlias.alias_normalized == normalized,
            )
        )
        if existing:
            return existing

        ea = EntityAlias(
            entity_id=entity.id,
            scope_key=scope_key,
            alias=alias.strip(),
            alias_normalized=normalized,
        )
        s.add(ea)
        return ea

    def in_scope(
        self,
        scope_key: str,
        *,
        entity_type: str | None = None,
        limit: int = 100,
        offset: int = 0,
        sa_session: SASession | None = None,
    ) -> list[Entity]:
        """List entities visible in scope (includes ancestor scopes up to global)."""
        with self._scope(sa_session) as s:
            stmt = select(Entity).where(build_scope_filter(scope_key, Entity.scope_key))
            if entity_type:
                stmt = stmt.where(Entity.entity_type == entity_type)
            stmt = stmt.order_by(Entity.created_at.desc()).limit(limit).offset(offset)
            entities = list(s.scalars(stmt))
            detach_all(entities, s)
            return entities

    def search(
        self,
        scope_key: str,
        query: str,
        *,
        entity_type: str | None = None,
        limit: int = 20,
        sa_session: SASession | None = None,
    ) -> list[Entity]:
        """Search entities by name/alias prefix within scope hierarchy."""
        pattern = self._like_pattern(query)
        with self._scope(sa_session) as s:
            stmt = (
                select(Entity)
                .join(EntityAlias, Entity.id == EntityAlias.entity_id)
                .where(
                    build_scope_filter(scope_key, EntityAlias.scope_key),
                    EntityAlias.alias_normalized.like(pattern, escape="\\"),
                )
            )
            if entity_type:
                stmt = stmt.where(Entity.entity_type == entity_type)
            entities = list(s.scalars(stmt.distinct().limit(limit)))
            detach_all(entities, s)
            return entities

    def _like_pattern(self, query: str) -> str:
        """Escape query for LIKE pattern matching."""
        q = query.lower().strip()
        escaped = q.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        return f"{escaped}%"

    @contextmanager
    def _scope(self, sa_session: SASession | None) -> Generator[SASession, None, None]:
        """Session scope - auto-commits on exit when we own the session."""
        if sa_session:
            yield sa_session
        else:
            with self._session_factory() as s:
                yield s
                s.commit()


class EntityRefStore:
    """Entity reference tracking - provenance and signal aggregation."""

    def __init__(self, lg: Logger, session_factory: SessionFactory) -> None:
        self._lg = lg
        self._session_factory = session_factory

    def add(
        self,
        entity_id: int,
        scope_key: str,
        source_type: str,
        *,
        source_id: str | None = None,
        source_url: str | None = None,
        snippet: str | None = None,
        sentiment: float | None = None,
        extra: dict[str, Any] | None = None,
        ref_at: datetime | None = None,
        sa_session: SASession | None = None,
    ) -> EntityRef:
        """Record a reference to an entity."""
        with self._scope(sa_session) as s:
            ref = EntityRef(
                entity_id=entity_id,
                scope_key=scope_key,
                source_type=source_type,
                source_id=source_id,
                source_url=source_url,
                snippet=snippet,
                sentiment=sentiment,
                extra=extra or {},
                ref_at=ref_at or datetime.now(UTC),
            )
            s.add(ref)
            s.flush()
            return cast(EntityRef, detach(ref, s))

    def count_by_entity(
        self,
        entity_id: int,
        *,
        scope_key: str | None = None,
        since: datetime | None = None,
        sa_session: SASession | None = None,
    ) -> int:
        """Count refs for an entity."""
        with self._scope(sa_session) as s:
            stmt = select(func.count(EntityRef.id)).where(EntityRef.entity_id == entity_id)
            if scope_key:
                stmt = stmt.where(build_scope_filter(scope_key, EntityRef.scope_key))
            if since:
                stmt = stmt.where(EntityRef.ref_at >= since)
            return s.scalar(stmt) or 0

    def recent_by_entity(
        self,
        entity_id: int,
        *,
        scope_key: str | None = None,
        limit: int = 10,
        sa_session: SASession | None = None,
    ) -> list[EntityRef]:
        """Get recent refs for an entity."""
        with self._scope(sa_session) as s:
            stmt = select(EntityRef).where(EntityRef.entity_id == entity_id)
            if scope_key:
                stmt = stmt.where(build_scope_filter(scope_key, EntityRef.scope_key))
            stmt = stmt.order_by(EntityRef.ref_at.desc()).limit(limit)
            refs = list(s.scalars(stmt))
            detach_all(refs, s)
            return refs

    def trending(
        self,
        scope_key: str,
        *,
        since: datetime | None = None,
        limit: int = 20,
        sa_session: SASession | None = None,
    ) -> list[tuple[Entity, int]]:
        """Get trending entities by ref count within scope."""
        with self._scope(sa_session) as s:
            stmt = (
                select(Entity, func.count(EntityRef.id).label("ref_count"))
                .join(EntityRef, Entity.id == EntityRef.entity_id)
                .where(
                    build_scope_filter(scope_key, EntityRef.scope_key),
                    build_scope_filter(scope_key, Entity.scope_key),
                )
            )
            if since:
                stmt = stmt.where(EntityRef.ref_at >= since)
            stmt = stmt.group_by(Entity.id).order_by(func.count(EntityRef.id).desc()).limit(limit)
            results = [(row[0], row[1]) for row in s.execute(stmt)]
            entities = [e for e, _ in results]
            detach_all(entities, s)
            return results

    @contextmanager
    def _scope(self, sa_session: SASession | None) -> Generator[SASession, None, None]:
        """Session scope - auto-commits on exit when we own the session."""
        if sa_session:
            yield sa_session
        else:
            with self._session_factory() as s:
                yield s
                s.commit()


class EntityRelationshipStore:
    """Entity-to-entity relationships within scoped subgraphs."""

    def __init__(self, lg: Logger, session_factory: SessionFactory) -> None:
        self._lg = lg
        self._session_factory = session_factory

    def add(
        self,
        from_entity_id: int,
        to_entity_id: int,
        relationship_type: str,
        scope_key: str,
        *,
        confidence: float = 1.0,
        extra: dict[str, Any] | None = None,
        sa_session: SASession | None = None,
    ) -> EntityRelationship:
        """Add a relationship between entities in a scope."""
        with self._scope(sa_session) as s:
            rel = EntityRelationship(
                from_entity_id=from_entity_id,
                to_entity_id=to_entity_id,
                relationship_type=relationship_type,
                scope_key=scope_key,
                confidence=confidence,
                extra=extra or {},
            )
            s.add(rel)
            s.flush()
            return cast(EntityRelationship, detach(rel, s))

    def get_relationships(
        self,
        entity_id: int,
        scope_key: str,
        *,
        direction: str = "both",
        relationship_type: str | None = None,
        sa_session: SASession | None = None,
    ) -> list[EntityRelationship]:
        """Get relationships for an entity within scope hierarchy."""
        if direction not in ("from", "to", "both"):
            raise ValueError(f"direction must be 'from', 'to', or 'both', got {direction!r}")

        with self._scope(sa_session) as s:
            dir_conditions = []
            if direction in ("from", "both"):
                dir_conditions.append(EntityRelationship.from_entity_id == entity_id)
            if direction in ("to", "both"):
                dir_conditions.append(EntityRelationship.to_entity_id == entity_id)

            stmt = select(EntityRelationship).where(
                or_(*dir_conditions),
                build_scope_filter(scope_key, EntityRelationship.scope_key),
            )

            if relationship_type:
                stmt = stmt.where(EntityRelationship.relationship_type == relationship_type)

            rels = list(s.scalars(stmt))
            detach_all(rels, s)
            return rels

    def get_relationships_for_entities(
        self,
        entity_ids: list[int],
        scope_key: str,
        *,
        direction: str = "both",
        relationship_type: str | None = None,
        sa_session: SASession | None = None,
    ) -> dict[int, list[EntityRelationship]]:
        """Get relationships for multiple entities. Returns {entity_id -> [relationships]}."""
        if not entity_ids:
            return {}
        if direction not in ("from", "to", "both"):
            raise ValueError(f"direction must be 'from', 'to', or 'both', got {direction!r}")

        with self._scope(sa_session) as s:
            stmt = self._build_batch_rel_query(entity_ids, scope_key, direction, relationship_type)
            results: dict[int, list[EntityRelationship]] = {eid: [] for eid in entity_ids}
            rels = list(s.scalars(stmt))
            for rel in rels:
                if direction in ("from", "both") and rel.from_entity_id in results:
                    results[rel.from_entity_id].append(rel)
                if direction in ("to", "both") and rel.to_entity_id in results:
                    if rel.to_entity_id != rel.from_entity_id or direction == "to":
                        results[rel.to_entity_id].append(rel)
            detach_all(rels, s)
            return results

    def _build_batch_rel_query(
        self,
        entity_ids: list[int],
        scope_key: str,
        direction: str,
        relationship_type: str | None,
    ) -> Any:
        """Build query for batch relationship lookup."""
        dir_conditions = []
        if direction in ("from", "both"):
            dir_conditions.append(EntityRelationship.from_entity_id.in_(entity_ids))
        if direction in ("to", "both"):
            dir_conditions.append(EntityRelationship.to_entity_id.in_(entity_ids))
        stmt = select(EntityRelationship).where(
            or_(*dir_conditions),
            build_scope_filter(scope_key, EntityRelationship.scope_key),
        )
        if relationship_type:
            stmt = stmt.where(EntityRelationship.relationship_type == relationship_type)
        return stmt

    @contextmanager
    def _scope(self, sa_session: SASession | None) -> Generator[SASession, None, None]:
        """Session scope - auto-commits on exit when we own the session."""
        if sa_session:
            yield sa_session
        else:
            with self._session_factory() as s:
                yield s
                s.commit()


class FactEntityStore:
    """Links between kelt atomic facts and KG entities."""

    def __init__(self, lg: Logger, session_factory: SessionFactory) -> None:
        self._lg = lg
        self._session_factory = session_factory

    def link(
        self,
        fact_id: int,
        entity_id: int,
        scope_key: str,
        *,
        role: str = "subject",
        confidence: float = 1.0,
        extra: dict[str, Any] | None = None,
        sa_session: SASession | None = None,
    ) -> FactEntity:
        """Link a fact to an entity."""
        with self._scope(sa_session) as s:
            fe = FactEntity(
                fact_id=fact_id,
                entity_id=entity_id,
                scope_key=scope_key,
                role=role,
                confidence=confidence,
                extra=extra or {},
            )
            s.add(fe)
            s.flush()
            return cast(FactEntity, detach(fe, s))

    def get_entities_for_fact(
        self,
        fact_id: int,
        scope_key: str,
        *,
        role: str | None = None,
        sa_session: SASession | None = None,
    ) -> list[tuple[Entity, str, float]]:
        """Get entities linked to a fact. Returns (entity, role, confidence)."""
        with self._scope(sa_session) as s:
            stmt = (
                select(Entity, FactEntity.role, FactEntity.confidence)
                .join(FactEntity, Entity.id == FactEntity.entity_id)
                .where(
                    FactEntity.fact_id == fact_id,
                    build_scope_filter(scope_key, FactEntity.scope_key),
                    build_scope_filter(scope_key, Entity.scope_key),
                )
            )
            if role:
                stmt = stmt.where(FactEntity.role == role)
            results = [(row[0], row[1], row[2]) for row in s.execute(stmt)]
            entities = [e for e, _, _ in results]
            detach_all(entities, s)
            return results

    def get_entities_for_facts(
        self,
        fact_ids: list[int],
        scope_key: str,
        *,
        role: str | None = None,
        entity_type: str | None = None,
        sa_session: SASession | None = None,
    ) -> dict[int, list[tuple[Entity, str, float]]]:
        """Get entities linked to multiple facts. Returns {fact_id -> [(entity, role, confidence), ...]}."""
        if not fact_ids:
            return {}

        with self._scope(sa_session) as s:
            stmt = (
                select(FactEntity.fact_id, Entity, FactEntity.role, FactEntity.confidence)
                .join(Entity, Entity.id == FactEntity.entity_id)
                .where(
                    FactEntity.fact_id.in_(fact_ids),
                    build_scope_filter(scope_key, FactEntity.scope_key),
                    build_scope_filter(scope_key, Entity.scope_key),
                )
            )
            if role:
                stmt = stmt.where(FactEntity.role == role)
            if entity_type:
                stmt = stmt.where(Entity.entity_type == entity_type)

            results: dict[int, list[tuple[Entity, str, float]]] = {fid: [] for fid in fact_ids}
            seen_entities: dict[int, Entity] = {}
            for row in s.execute(stmt):
                fact_id, entity, r, conf = row[0], row[1], row[2], row[3]
                results[fact_id].append((entity, r, conf))
                seen_entities[entity.id] = entity

            detach_all(list(seen_entities.values()), s)
            return results

    def get_facts_for_entity(
        self,
        entity_id: int,
        scope_key: str,
        *,
        role: str | None = None,
        limit: int = 100,
        sa_session: SASession | None = None,
    ) -> list[int]:
        """Get fact IDs linked to an entity."""
        with self._scope(sa_session) as s:
            stmt = select(FactEntity.fact_id).where(
                FactEntity.entity_id == entity_id,
                build_scope_filter(scope_key, FactEntity.scope_key),
            )
            if role:
                stmt = stmt.where(FactEntity.role == role)
            stmt = stmt.limit(limit)
            return list(s.scalars(stmt))

    @contextmanager
    def _scope(self, sa_session: SASession | None) -> Generator[SASession, None, None]:
        """Session scope - auto-commits on exit when we own the session."""
        if sa_session:
            yield sa_session
        else:
            with self._session_factory() as s:
                yield s
                s.commit()


class KGStore:
    """Unified facade for all KG operations."""

    def __init__(
        self,
        lg: Logger,
        session_factory: SessionFactory,
        embedder: EmbeddingClient | None = None,
        embedding_factory: EmbeddingFactory | None = None,
        embedding_format: QuantizationFormat | None = None,
    ) -> None:
        self._lg = lg
        self._session_factory = session_factory
        self._embedder = embedder
        self._embedding_factory = embedding_factory
        self._embedding_format = embedding_format or QuantizationFormat.F16

        self.entities = EntityStore(lg, session_factory)
        self.refs = EntityRefStore(lg, session_factory)
        self.relationships = EntityRelationshipStore(lg, session_factory)
        self.fact_entities = FactEntityStore(lg, session_factory)

        # Embedding adapter (only if factory provided)
        self._embedding_adapter: EntityEmbeddingAdapter | None = None
        if embedding_factory is not None:
            from .embedding import EntityEmbeddingAdapter

            self._embedding_adapter = EntityEmbeddingAdapter(
                session_factory=session_factory,
                factory=embedding_factory,
                format=self._embedding_format,
                embedder=embedder,
            )

    @property
    def embeddings(self) -> EntityEmbeddingAdapter:
        """Access entity embedding operations.

        Raises:
            RuntimeError: If embeddings client was not provided at construction.
        """
        if self._embedding_adapter is None:
            raise RuntimeError("Embeddings not configured: embeddings client not provided")
        return self._embedding_adapter
