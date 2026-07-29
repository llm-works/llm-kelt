# Multi-schema and isolation

Two orthogonal isolation mechanisms:

- **`context_key`** — a string filter on every atomic read/write. Cheap; everything shares
  one Postgres schema. Enough for most single-tenant applications.
- **`schema_name`** — a distinct Postgres schema per tenant. Every table exists per-schema.
  Genuine physical separation; different tenants can even be on different Postgres
  extensions or migration versions.

Plus one control:

- **`SchemaMode`** — what the `Client` does to the schema on startup (create it, verify it's
  current, or leave it alone).

## `SchemaMode`

```python
from llm_kelt import SchemaMode

# SchemaMode values:
# SchemaMode.ENSURE  — run migrations to head (default)
# SchemaMode.VERIFY  — check schema is at head; raise SchemaVersionError otherwise
# SchemaMode.SKIP    — don't touch alembic
```

Pass at client construction:

```python
kelt = ClientFactory(lg).create_from_config(
    context=ClientContext(context_key="my-agent"),
    config=config,
    schema_mode=SchemaMode.ENSURE,
)
```

### When to use each

`ENSURE` — the default. A single writer that also owns migrations. Runs alembic on startup,
creates the schema if missing, creates the `vector` extension if missing. Idempotent and
thread-safe (Postgres advisory lock; version table
`alembic_version_kelt` per schema).

`VERIFY` — read-heavy services in a fleet where one dedicated writer runs migrations. Each
`VERIFY` client checks that the schema on disk matches the version the library expects; if
not, raises `SchemaVersionError` and refuses to start. Prevents a stale service from writing
to a schema that has already been upgraded past it.

```python
try:
    client = ClientFactory(lg).create_from_config(
        context=ctx,
        config=config,
        schema_mode=SchemaMode.VERIFY,
    )
except SchemaVersionError as e:
    lg.fatal("schema drift", extra={"exception": e})
    raise
```

`SKIP` — read-only consumers that must not import pgvector or numpy (lightweight sidecars,
web dashboards). Doesn't run alembic, doesn't validate anything, and skips creating the
default embedding store — so nothing pulls in the heavyweight vector dependencies. Trying to
touch `.embeddings` on a SKIP client still raises `RuntimeError` unless an explicit
`EmbeddingStoreClient` was passed in.

`get_schema_status()` on a SKIP client also raises `RuntimeError` — the whole point of SKIP
is that the alembic modules are never imported.

## Isolation vs schema — pick the axis

Two flat axes, four combinations. Pick per-write, not per-service.

| context_key | schema | Effect |
|---|---|---|
| One per tenant | Shared | Cheap. One migration. All tenants in one schema. Standard for SaaS with light isolation needs. |
| Shared | One per tenant | Physical separation. Independent migrations per tenant. Heavier but supports diverging schemas. |
| One per tenant | One per tenant | Maximum isolation; typically overkill. |
| Shared | Shared | Only when there is genuinely one tenant. |

## `ClientContext`

```python
@dataclass
class ClientContext:
    context_key: str | None = None  # supports * and ? SQL LIKE globs on reads
    schema_name: str | None = None  # None → "public"
```

Both fields default to `None`. A `None` `context_key` on the client means "read across every
key" — useful for admin/analytics tools.

### Globbed reads

```python
admin = ClientFactory(lg).create_from_config(
    context=ClientContext(context_key="acme:*"),  # read every acme:* key
    config=config,
    schema_mode=SchemaMode.VERIFY,
)

for fact in admin.atomic.assertions.list_active(limit=1000):
    print(fact.context_key, fact.content)
```

Writes with a glob key raise — you can't write to `"*"`. Construct a separate isolated
writer per real key, or use `with_isolation()`.

## `with_isolation` — new client, different context

Copy the current client's dependencies (database, embedder, LLM client) but override the
isolation key:

```python
planner = kelt.with_isolation(context_key="agent:planner")
reviewer = kelt.with_isolation(context_key="agent:reviewer")

planner.atomic.assertions.add("Always draft before executing")
reviewer.atomic.assertions.add("Never approve without reading tests")
```

The result is a full `Client` — `.atomic`, `.query`, `.train`, everything works. Signature:

```python
def with_isolation(self, *, context_key=..., schema_name=..., **kwargs) -> Client
```

Schema mode downgrades on isolation: `ENSURE → VERIFY`, `SKIP → SKIP`. The child client
won't re-run migrations on the parent's schema.

## `with_schema` — scoped handle, lazy schema init

`ScopedClient` is a lightweight handle that points at a different Postgres schema. On first
`.atomic` access, it initialises the schema (creates it, runs migrations) under an advisory
lock so concurrent first-uses converge.

```python
prod = kelt.with_schema("production")
stage = kelt.with_schema("staging")

prod.atomic.assertions.add("...", category="config")   # writes to production.memv1_facts
stage.atomic.feedback.record(signal="positive", ...)   # writes to staging.memv1_feedback_details
```

Signature:

```python
def with_schema(self, schema_name: str) -> ScopedClient
```

`ScopedClient` currently exposes `.atomic` and `.schema_name`. For anything beyond atomic
memory, use `with_isolation(schema_name=...)` instead — that returns a full `Client`.

### When to prefer `with_schema` over `with_isolation`

- Short-lived writes to another schema (one loop, one job). Cheaper: no full client rebuild.
- Training data pipelines reading from many schemas:

```python
for schema in ["prod-east", "prod-west", "staging"]:
    scoped = kelt.with_schema(schema)
    for pref in scoped.atomic.preferences.list(limit=10000):
        yield pref
```

### When to prefer `with_isolation(schema_name=...)`

- You need the full client surface (`.query`, `.train`, `.kg`, `.content`).
- You want to override `context_key` at the same time.

```python
tenant_client = kelt.with_isolation(
    context_key="tenant:acme:agent:reviewer",
    schema_name="tenant_acme",
)
answer = await tenant_client.query.ask("...", rag=RAGArgs(top_k=5))
```

## Schema validation and diagnostics

```python
status = kelt.get_schema_status()
# → SchemaStatus(state=SchemaState.CURRENT, current_revision="a1b2c3", head_revision="a1b2c3")
```

`SchemaState`:

- `MISSING` — schema doesn't exist. `ENSURE` would create it.
- `CURRENT` — matches library expectation.
- `NEEDS_UPGRADE` — schema exists but is behind. `ENSURE` runs migrations to head; `VERIFY`
  raises.
- `TOO_NEW` — schema is ahead of what this library version knows. Always raises. Fix by
  upgrading the library.

`get_schema_status()` on a SKIP client raises `RuntimeError` — SKIP means alembic modules
weren't imported at all.

## Cross-schema training data

Training manifests can reference a source schema in their `source` section, and the runner
reads training data from that schema regardless of what the caller was scoped to:

```yaml
# manifests/prod-sft.yaml
adapter: prod-sft-v1
method: sft
source:
  schema_name: production
  context_key: agent:code-reviewer
data:
  format: external
  path: ./exports/prod_positive.jsonl
```

If you're exporting programmatically:

```python
scoped = kelt.with_schema("production")
result = export_feedback_sft(
    session_factory=scoped.database.session,
    context_key="agent:code-reviewer",
    output_path="prod_positive.jsonl",
    signal="positive",
)
```

## Schema naming

Validated against `^[a-z][a-z0-9_]{0,62}$`. Lower-case, starts with a letter, ≤63 chars.
Rejecting anything else prevents `search_path` injection.

The `public` schema is the default when `schema_name` is `None`.

## Migration internals (in one paragraph)

Each schema has its own `alembic_version_kelt` version table (namespaced in #84 to avoid
colliding with other libraries doing alembic in the same schema). Migrations run under a
fixed Postgres advisory lock (`_ADVISORY_LOCK_KEY = 7829104563218907456`) so concurrent
`ENSURE` clients converge to one migration run. Migrations create the schema if missing and
run `CREATE EXTENSION IF NOT EXISTS vector`, so pgvector must be available on the server.
