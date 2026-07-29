# Knowledge graph

Entity-centric storage: named things (companies, projects, models, contracts) with aliases,
relationships, and per-entity ref feeds. Sits alongside atomic memory — same database, same
`Client`, different mental model.

Use this when your domain has real named entities you want to reference by identity across
many facts. Otherwise stick with atomic assertions.

```python
kg = kelt.kg  # KGStore instance
```

## Scope keys — hierarchical

Every KG record has a `scope_key: str`. Reads resolve *up* the hierarchy; writes always land
in the exact scope you specified.

```text
"org:acme:user:alice"
       ↓ read visibility
"org:acme:user:alice"   ← the record
"org:acme"              ← inherited
"global"                ← inherited
```

Convention: colon-separated `key:value` pairs. `"global"` is the root. `scope_ancestors()`
enumerates the resolved chain:

```python
from llm_kelt.memory.kg.store import scope_ancestors

scope_ancestors("org:acme:user:alice")
# → ["org:acme:user:alice", "org:acme", "global"]
```

This is **not** the same as the atomic `context_key`. `context_key` is a flat isolation
key filtered exactly on every read. `scope_key` is hierarchical — narrower scopes see
broader ones.

## Entities — `kg.entities`

```python
class EntityStore:
    def get(self, entity_id: int) -> Entity | None
    def get_by_name(self, scope_key, name, entity_type) -> Entity | None
    def get_by_names(self, scope_key, names: list[str], entity_type) -> list[Entity]
    def create(
        self, scope_key: str, canonical_name: str, entity_type: str, *,
        description: str | None = None,
        extra: dict | None = None,
        aliases: list[str] | None = None,
    ) -> Entity
    def find_or_create(
        self, scope_key: str, name: str, entity_type: str, *,
        description=None, extra=None,
    ) -> tuple[int, bool]                              # (entity_id, created)
    def update(self, entity_id, *, description=None, extra=None) -> Entity | None
    def delete(self, entity_id: int) -> bool
    def delete_in_scope(self, scope_key: str, *, entity_type: str | None = None) -> int
    def resolve(self, scope_key, name, entity_type=None) -> Entity | None
    def add_alias(self, entity_id, alias, scope_key) -> EntityAlias | None
    def in_scope(self, scope_key, *, entity_type=None, limit=100, offset=0) -> list[Entity]
    def search(self, scope_key, query, *, entity_type=None, limit=20) -> list[Entity]
```

Names are stored lowercased and stripped as `canonical_name`. Match is case-insensitive.

### Create with aliases

```python
tesla = kg.entities.create(
    scope_key="global",
    canonical_name="Tesla",
    entity_type="company",
    description="Electric vehicle manufacturer",
    aliases=["TSLA", "Tesla Motors", "Tesla, Inc."],
)
print(tesla.id, tesla.canonical_name, [a.alias for a in tesla.aliases])
```

### Idempotent lookup

```python
eid, created = kg.entities.find_or_create(
    scope_key="global",
    name="OpenAI",
    entity_type="company",
    description="AI research and deployment company",
)
```

`find_or_create` first resolves via alias, then via canonical name up the scope hierarchy;
only creates if neither hit. `create` inside a nested transaction, so a concurrent race
converges on one row.

### Resolve by any alias

```python
found = kg.entities.resolve(
    scope_key="org:acme",
    name="TSLA",  # or "Tesla" or "Tesla, Inc."
    entity_type="company",
)
```

Resolution walks the scope hierarchy — an alias created in `"global"` matches a lookup in
`"org:acme:user:alice"`.

### Add an alias later

```python
result = kg.entities.add_alias(tesla.id, "$TSLA", scope_key="global")
if result is None:
    print("alias already owned by a different entity")
```

Returns `None` on conflict rather than raising — mirror of `find_or_create` semantics.

### Delete

```python
kg.entities.delete(tesla.id)  # single, cascades to aliases/refs/relationships
kg.entities.delete_in_scope("org:acme:session:xyz")  # bulk delete of a whole scope, exact match
```

`delete_in_scope` deliberately does *not* walk ancestors — a bulk delete of the exact scope
you passed, so global entities are safe.

### List and search

```python
kg.entities.in_scope("org:acme", entity_type="company", limit=50)
kg.entities.search("org:acme", "tesl", entity_type="company")  # prefix on aliases
```

## Alias conflicts

```python
from llm_kelt.memory.kg import AliasConflictError

try:
    kg.entities.create(scope_key="global", canonical_name="Tesla", entity_type="company")
except AliasConflictError as e:
    print(e.alias, "already owned by", e.entity_id)
```

`create` raises `AliasConflictError` if its own canonical name is already an alias for a
different entity in the same scope. `add_alias` returns `None` in the same situation.

## Refs — `kg.refs`

Record every mention of an entity — source, timestamp, snippet, sentiment. Cheap to write,
useful for provenance and "trending" queries.

```python
def add(
    entity_id: int, scope_key: str, source_type: str, *,
    source_id: str | None = None, source_url: str | None = None,
    snippet: str | None = None,
    sentiment: float | None = None,
    extra: dict | None = None,
    ref_at: datetime | None = None,
) -> EntityRef
```

Example:

```python
from datetime import datetime, UTC

kg.refs.add(
    entity_id=tesla.id,
    scope_key="org:acme:analyst:bob",
    source_type="article",
    source_id="reuters-2026-07-15",
    source_url="https://…",
    snippet="Tesla reported record Q2 deliveries…",
    sentiment=0.6,
)

kg.refs.count_by_entity(tesla.id, since=datetime(2026, 1, 1, tzinfo=UTC))
kg.refs.recent_by_entity(tesla.id, scope_key="org:acme", limit=10)

# Which entities are getting the most mentions in this scope?
trending = kg.refs.trending("org:acme", since=datetime(2026, 6, 1, tzinfo=UTC), limit=20)
for entity, ref_count in trending:
    print(ref_count, entity.canonical_name)
```

## Relationships — `kg.relationships`

Typed, directional edges between two entities in a scope.

```python
def add(
    from_entity_id: int, to_entity_id: int,
    relationship_type: str, scope_key: str,
    *, confidence: float = 1.0, extra: dict | None = None,
) -> EntityRelationship

def get_relationships(
    entity_id: int, scope_key: str, *,
    direction: Literal["from", "to", "both"] = "both",
    relationship_type: str | None = None,
) -> list[EntityRelationship]

def get_relationships_for_entities(
    entity_ids: list[int], scope_key: str,
    *, direction="both", relationship_type=None,
) -> dict[int, list[EntityRelationship]]
```

Example:

```python
elon_id, _ = kg.entities.find_or_create("global", "Elon Musk", "person")
spacex_id, _ = kg.entities.find_or_create("global", "SpaceX", "company")

kg.relationships.add(elon_id, tesla.id, "founded", scope_key="global", confidence=0.9)
kg.relationships.add(elon_id, spacex_id, "founded", scope_key="global")
kg.relationships.add(tesla.id, spacex_id, "related_to", scope_key="org:acme")

# All edges touching Elon
for rel in kg.relationships.get_relationships(elon_id, "global"):
    print(rel.from_entity_id, rel.relationship_type, rel.to_entity_id)

# Only outgoing "founded" edges
kg.relationships.get_relationships(
    elon_id,
    "global",
    direction="from",
    relationship_type="founded",
)
```

Batch query — pull all edges for a set of entities in one round-trip:

```python
by_entity = kg.relationships.get_relationships_for_entities(
    entity_ids=[elon_id, tesla.id, spacex_id],
    scope_key="global",
)
# → {entity_id: [EntityRelationship, ...]}
```

## Linking facts to entities — `kg.fact_entities`

Bridge between the atomic side and the KG side. Every atomic fact stays a first-class row;
`fact_entities` adds an edge saying "this fact talks about this entity".

```python
def link(
    fact_id: int, entity_id: int, scope_key: str, *,
    role: str = "subject",
    confidence: float = 1.0,
    extra: dict | None = None,
    if_not_exists: bool = False,
) -> FactEntity | None

def get_entities_for_fact(fact_id, scope_key, *, role=None) -> list[tuple[Entity, str, float]]
def get_entities_for_facts(fact_ids: list[int], scope_key, *, role=None, entity_type=None) -> dict[int, list[tuple[Entity, str, float]]]
def get_facts_for_entity(entity_id, scope_key, *, role=None, limit=100) -> list[int]
```

Example:

```python
fid = kelt.atomic.assertions.add(
    "Tesla reported record Q2 deliveries",
    category="market",
    source="reuters-2026-07-15",
)
kg.fact_entities.link(fid, tesla.id, scope_key="org:acme", role="subject")

# All entities mentioned in fact fid
for entity, role, conf in kg.fact_entities.get_entities_for_fact(fid, "org:acme"):
    print(role, entity.canonical_name, conf)

# All facts mentioning Tesla in this scope
fact_ids = kg.fact_entities.get_facts_for_entity(tesla.id, "org:acme")
facts = kelt.atomic.assertions.get_many(fact_ids)
```

`if_not_exists=True` does a Postgres upsert (`ON CONFLICT DO NOTHING`) and returns `None` if
the link already exists — safe to call in a loop without wrapping IntegrityError.

Batch pull entities for a page of facts:

```python
mapping = kg.fact_entities.get_entities_for_facts(
    fact_ids=[f.id for f in page_of_facts],
    scope_key="org:acme",
    entity_type="company",
)
# → {fact_id: [(entity, role, confidence), ...]}
```

## Entity embeddings — `kg.embeddings`

Available when the `Client` was constructed with an `embedding_factory`. Same underlying
store as atomic embeddings, but keyed on `entity_id` under `entity_type="kg.entity"`.

```python
kg.embeddings.set_embedding(entity_id, [0.12, -0.03, ...], model="text-embedding-3-small")
kg.embeddings.search_similar(
    query=query_vector,
    model="text-embedding-3-small",
    top_k=10,
    min_similarity=0.4,
)
```

`RuntimeError` if the embedding factory wasn't provided at `Client` construction.

## Choosing atomic vs KG

| Question | Answer → use |
|---|---|
| "Does the domain have named, canonical things I need to dedupe by identity?" | KG entities |
| "Do I need to walk relationships between things?" | KG relationships |
| "Am I recording free-text notes with categories?" | Atomic assertions |
| "Am I tracking feedback/preferences/predictions?" | Atomic (feedback/preferences/predictions) |
| "Do I want scope inheritance (child scope sees parent)?" | KG scope_key |
| "Do I want strict per-tenant isolation?" | Atomic context_key |
| "Both — facts *about* entities?" | Both — link with `kg.fact_entities.link()` |

Nothing prevents you from using only atomic memory. The KG side is opt-in and adds tables
(`kg_entities`, `kg_aliases`, `kg_refs`, `kg_relationships`, `kg_fact_entities`) that stay
empty until you write to them.
