# Concepts

Read once. The tutorials assume these terms.

## Context key

Every record belongs to a `context_key: str`. Reads are filtered by it. It's the isolation
boundary — one agent, one user, one tenant, whatever unit you need. Set on the client:

```python
kelt = factory.create_from_config(
    context=ClientContext(context_key="agent:code-reviewer"),
    config=config,
)
```

`context_key` supports SQL `LIKE` globs on read paths — `"acme:*"` or `"*"` — so a client
constructed with a glob key can read across many isolated writers without merging their
writes. Writes always use the literal string.

Switch to a different key without rebuilding the client:

```python
sub = kelt.with_isolation(context_key="agent:planner")
```

## Schema

Isolation-key filtering happens *inside* one Postgres schema. Different schemas hold different
copies of every table — genuine physical separation. Pick a schema by setting `schema_name`
on the context, or by opening a `ScopedClient`:

```python
prod = kelt.with_schema("production")
stage = kelt.with_schema("staging")
prod.atomic.assertions.add("...")  # writes to production.memv1_facts
```

Schema-scoped clients are lazily initialised on first `.atomic` access. Use them when
different logical tenants share a database but must not see each other's data.

See [multi-schema](multi-schema.md) for `SchemaMode` (ENSURE / VERIFY / SKIP) and how the
library manages migrations.

## Atomic memory vs knowledge graph

Two orthogonal storage subsystems live under one `Client`:

- **Atomic memory** (`kelt.atomic.*`) — a bag of typed *facts*. Every write goes into a shared
  `Fact` row plus one details row (`FeedbackDetails`, `PreferenceDetails`, …) keyed to the
  fact. Reads are per-type: `assertions.list_active()`, `feedback.list_by_signal("positive")`.
- **Knowledge graph** (`kelt.kg`) — named *entities* with hierarchical *scope keys*, aliases,
  and relationships between entities. Facts can be linked to entities via `fact_entities`.

The atomic side is enough for most memory/RAG/training work. The KG side is for when your
domain has real named things (organisations, projects, models, contracts) that you want to
reference by identity across many facts. See [knowledge graph](knowledge-graph.md).

## The seven atomic clients

All share the `FactClient` base (`get`, `list`, `count`, `delete`, `activate`, `deactivate`,
`exists`). Each adds a typed `add()` or `record()` that writes into its details table:

| Client | Method | Purpose |
|---|---|---|
| `atomic.assertions` | `add(content, category=)` | Free-text statements to inject into prompts |
| `atomic.feedback` | `record(signal, content_id=, strength=, tags=)` | Positive/negative/dismiss signals on content |
| `atomic.preferences` | `record(context, chosen, rejected, margin=)` | DPO training pairs |
| `atomic.predictions` | `record(hypothesis, confidence, resolution_date=)` | Track hypotheses; resolve later |
| `atomic.directives` | `record(text, directive_type=, expires_at=)` | Standing / one-time / rule directives |
| `atomic.interactions` | `record(interaction_type, content_id=, duration_ms=)` | Implicit signals (view, click, read) |
| `atomic.solutions` | `record(agent_name, problem, problem_context, answer, ...)` | Agent runs — question, tool calls, answer |

Two more sit alongside them:

- `atomic.relationships` — typed edges between facts (`derived_from`, `contradicts`,
  `supports`, …).
- `atomic.embeddings` — vector embeddings for facts. Used for RAG.

Details, signatures, and worked examples: [atomic memory](atomic-memory.md).

## Content

`kelt.content.create(content_text, source=, title=)` returns a `content_id: int` for a
deduplicated content row. Feedback, interactions, and other clients reference content by
this id instead of copying the text.

```python
cid = kelt.content.create(content_text="…the LLM's answer…", source="agent")
kelt.atomic.feedback.record(signal="positive", content_id=cid, strength=0.9)
```

## Client hierarchy

```
ClientFactory        ← constructed once, holds the logger
  └── Client         ← constructed per (context_key, schema, config)
        ├── .atomic          → seven clients + relationships + embeddings
        ├── .kg              → KGStore
        ├── .content         → ContentStore
        ├── .train           → training subsystem
        ├── .query           → ContextQuery (lazy, requires llm_client)
        ├── .context_builder → ContextBuilder over atomic.assertions
        ├── .database        → SQLAlchemy engine + session factory
        └── .with_schema()   → returns ScopedClient for a different schema
        └── .with_isolation()→ returns Client with overridden context_key/schema
```

The factory constructs `Client` from a config file section, but you can also construct a
`Client` directly with an existing `Database`, `EmbeddingClient`, and `ChatClient` if you
already have them wired up in the surrounding application.

## Errors

Every library-level exception is a `KeltError` subclass. Import from top-level:

```python
from llm_kelt import (
    KeltError,
    ValidationError,
    NotFoundError,
    DatabaseError,
    ConfigError,
    ConflictError,
    SchemaVersionError,
)
```

- `ValidationError` — bad argument (empty content, out-of-range confidence, unknown category
  filter).
- `NotFoundError` — `predictions.resolve(fact_id, ...)` on a missing or non-prediction fact.
- `ConflictError` — `relationships.link(...)` on a duplicate edge.
- `SchemaVersionError` — schema on disk is a different version than the library expects;
  raised on `VERIFY` mode or when downgrade is detected.
- `DatabaseError`, `ConfigError` — reserved for library-level infra failures.
- `ContextOverflowError` — `Conversation.append` when a single message can't fit even after
  compaction. See [conversation](conversation.md).

## Config

The library reads `config.dbs.<db_key>` (default `db_key="main"`) and, when the relevant
subsystem is used, `config.llm`, `config.embedding`, `config.kelt`, and `config.training`.
`Config` is `appinfra.config.Config` — a nested `DotDict` loaded from YAML with `!path` and
`!env` tags.

Nothing is validated at construction. If a section is missing when a subsystem is accessed
(e.g. `client.train` without `adapters.lora.base_path`), the property raises a `RuntimeError`.
