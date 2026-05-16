"""Knowledge Graph layer for entity-centric knowledge management.

Extends kelt's atomic facts with:
- Canonical entities with identity-based deduplication
- Scoped subgraphs via hierarchical scope keys
- Entity resolution via aliases
- Entity relationships
- Fact-entity linkage

Usage::

    from llm_kelt.memory.kg import KGStore

    kg = KGStore(lg, session_factory)

    # Find or create entity
    tesla_id, created = kg.entities.find_or_create(
        scope_key="global",
        name="Tesla",
        entity_type="company",
    )

    # Add alias for dedup
    kg.entities.add_alias(tesla_id, "TSLA", scope_key="global")

    # Link kelt fact to entity
    kg.fact_entities.link(fact_id=42, entity_id=tesla_id, scope_key="org:acme")

    # Query with scope (sees scope + ancestors up to global)
    entities = kg.entities.in_scope("org:acme:user:alice")
"""

from .embedding import EntityEmbeddingAdapter
from .models import (
    Entity,
    EntityAlias,
    EntityRef,
    EntityRelationship,
    FactEntity,
)
from .store import (
    EntityRefStore,
    EntityRelationshipStore,
    EntityStore,
    FactEntityStore,
    KGStore,
)

__all__ = [
    # Models
    "Entity",
    "EntityAlias",
    "EntityRef",
    "EntityRelationship",
    "FactEntity",
    # Stores
    "KGStore",
    "EntityStore",
    "EntityRefStore",
    "EntityRelationshipStore",
    "FactEntityStore",
    # Embedding
    "EntityEmbeddingAdapter",
]
