# Atomic memory

Seven fact clients, one shared `Fact` row per record, plus a per-type details row. All hang
off `kelt.atomic`. Every method here writes or reads under the current `context_key`.

Setup used throughout this doc:

```python
from datetime import datetime, UTC

from appinfra.config import Config
from appinfra.log import LogConfig, LoggerFactory
from llm_kelt import ClientContext, ClientFactory

config = Config("etc/llm-kelt.yaml")
lg = LoggerFactory.create_root(LogConfig.from_params(level="warning"))
kelt = ClientFactory(lg).create_from_config(
    context=ClientContext(context_key="tutorial:atomic"),
    config=config,
)
```

## The `Fact` row

Every atomic write returns an `int` fact ID pointing at a row in `memv1_facts`:

| Column | Type | Notes |
|---|---|---|
| `id` | int | Primary key |
| `context_key` | str | Isolation key |
| `type` | str | `"assertion"`, `"feedback"`, `"preference"`, `"prediction"`, `"directive"`, `"interaction"`, `"solution"` |
| `content` | str | Human-readable summary (varies by type) |
| `category` | str? | Free-text tag |
| `source` | str | Where it came from (`"user"`, `"agent"`, ...) |
| `confidence` | float | 0.0–1.0 |
| `active` | bool | `false` = soft-deleted |
| `created_at`, `updated_at` | datetime | UTC |

Type-specific fields live in a details table joined by `fact_id` (`memv1_feedback_details`,
`memv1_preference_details`, ...). You rarely touch those directly — the client methods
return `Fact` objects with the details eagerly loaded.

## Common methods

Every client inherits these from `FactClient`:

```python
client.get(fact_id)                                → Fact | None
client.list(limit=100, offset=0, active_only=True) → list[Fact]
client.count(active_only=True)                     → int
client.exists(fact_id)                             → bool
client.activate(fact_id)                           → bool
client.deactivate(fact_id)                         → bool       # soft delete
client.delete(fact_ids)                            → DeleteResult  # hard delete (single id or iterable)
```

Type-specific `add` / `record` methods return the new `fact_id`.

---

## Assertions — `kelt.atomic.assertions`

Free-text statements. The primary source for `ContextBuilder` prompt injection.

```python
def add(
    content: str,
    category: str | None = None,
    source: str = "user",
    confidence: float = 1.0,
    *,
    context: dict[str, Any] | None = None,
) -> int
```

Example:

```python
kelt.atomic.assertions.add("Prefers concise, code-first answers", category="style")
kelt.atomic.assertions.add(
    "API errors use {code, message, request_id}",
    category="api",
    confidence=0.9,
    source="onboarding-doc",
)
```

Query methods beyond the shared ones:

```python
list_active(category=None, min_confidence=0.0, limit=100)  → list[Fact]
list_by_category(category, limit=100, active_only=True)    → list[Fact]
list_by_source(source, limit=100, active_only=True)        → list[Fact]
search(query, limit=50, active_only=True)                  → list[Fact]  # ILIKE %query%
get_categories()                                           → list[str]
count_by_category()                                        → dict[str | None, int]
get_many(fact_ids)                                         → list[Fact]
update(fact_id, content=None, category=None, confidence=None) → bool
```

`get_many` is a single SQL query — use it in place of a loop of `get()`.

## Feedback — `kelt.atomic.feedback`

Explicit signals on content items.

```python
def record(
    signal: Literal["positive", "negative", "dismiss"],
    content_id: int | None = None,
    strength: float = 1.0,
    tags: list[str] | None = None,
    comment: str | None = None,
    context: dict | None = None,
    category: str | None = None,
    provider_type: str | None = None,
    provider: str | None = None,
    feedback_at: datetime | None = None,
) -> int
```

`content_id` is optional but usual — it's how you tie feedback to a stored content row:

```python
cid = kelt.content.create(content_text="…the answer…", source="agent")

kelt.atomic.feedback.record(
    signal="positive",
    content_id=cid,
    strength=0.9,
    tags=["clear", "concise"],
)
```

Query methods:

```python
list_by_signal(signal, limit=100, active_only=True) → list[Fact]
list_by_content(content_id, limit=100)              → list[Fact]
count_by_signal()                                   → dict[str, int]
```

Feedback is the input to the SFT and classifier training exports. See [training](training.md).

## Preferences — `kelt.atomic.preferences`

Preference pairs for DPO training.

```python
def record(
    context: str,
    chosen: str,
    rejected: str,
    margin: float | None = None,
    category: str | None = None,
    extra: dict | None = None,
) -> int
```

Example:

```python
kelt.atomic.preferences.record(
    context="Explain backpropagation",
    chosen="Backprop applies the chain rule backward through layers to compute gradients.",
    rejected="Backprop is a complex algorithm involving lots of derivatives and stuff.",
    margin=0.8,
    category="ml_explanations",
)
```

Query methods:

```python
list_by_category(category, limit=100, active_only=True) → list[Fact]
get_categories()                                        → list[str]
search(query, limit=50, active_only=True)               → list[Fact]  # ILIKE over context
```

DPO export → [training](training.md).

## Predictions — `kelt.atomic.predictions`

Hypotheses recorded now, resolved later. Useful for calibration tracking.

```python
def record(
    hypothesis: str,
    confidence: float,                                  # 0.0–1.0
    resolution_date: date | str | None = None,
    resolution_event: str | None = None,
    resolution_metric: dict | None = None,
    category: str | None = None,
    tags: list[str] | None = None,
    verification_source: str | None = None,
    verification_url: str | None = None,
    *,
    context: dict[str, Any] | None = None,
) -> int

def resolve(
    fact_id: int,
    outcome: Literal["correct", "incorrect", "partial", "cancelled"],
    actual: str | None = None,
    outcome_confidence: float | None = None,
) -> bool
```

Example:

```python
pid = kelt.atomic.predictions.record(
    hypothesis="Model X will outperform baseline by ≥3% F1",
    confidence=0.7,
    resolution_date="2026-08-01",
    resolution_event="benchmark_v2 complete",
    category="ml_experiments",
)

# Later:
kelt.atomic.predictions.resolve(pid, outcome="correct", actual="4.2% F1 gain")
```

Query methods:

```python
list_pending(category=None, limit=100)                             → list[Fact]
list_resolved(outcome=None, category=None, since=None, limit=100)  → list[Fact]
get_calibration_stats(category=None)                               → dict[str, dict]
```

`get_calibration_stats` buckets predictions by their stated confidence and reports the
observed correct-rate per bucket — the input to a reliability diagram.

## Directives — `kelt.atomic.directives`

Rules, standing instructions, one-time tasks.

```python
def record(
    text: str,
    directive_type: Literal["standing", "one-time", "rule"] = "standing",
    parsed_rules: dict | None = None,
    expires_at: datetime | None = None,
    category: str | None = None,
    *,
    context: dict | None = None,
) -> int

def set_status(fact_id: int, status: Literal["active", "paused", "completed"]) → bool
```

Example:

```python
kelt.atomic.directives.record(
    text="Never commit to main without a green CI run",
    directive_type="rule",
    category="git",
)

did = kelt.atomic.directives.record(
    text="Prepare v0.4.0 release notes by Friday",
    directive_type="one-time",
    expires_at=datetime(2026, 8, 1, tzinfo=UTC),
)
kelt.atomic.directives.set_status(did, "completed")
```

Query methods:

```python
list_active(directive_type=None, limit=100)                    → list[Fact]
list_by_type(directive_type, include_inactive=False, limit=100) → list[Fact]
```

## Interactions — `kelt.atomic.interactions`

Implicit signals on content.

```python
def record(
    interaction_type: Literal["view", "click", "read", "scroll", "dismiss"],
    content_id: int | None = None,
    duration_ms: int | None = None,
    scroll_depth: float | None = None,
    context: dict | None = None,
    category: str | None = None,
) -> int
```

Example:

```python
kelt.atomic.interactions.record(
    interaction_type="read",
    content_id=cid,
    duration_ms=12500,
    scroll_depth=0.87,
)
```

Query methods:

```python
list_by_type(interaction_type, limit=100)   → list[Fact]
list_by_content(content_id, limit=100)      → list[Fact]
count_by_type()                             → dict[str, int]
get_engagement_stats(content_id)            → dict[str, int | float | None]
```

`get_engagement_stats` returns view count, total dwell, mean scroll depth, dismiss rate.

## Solutions — `kelt.atomic.solutions`

Recorded agent runs. Store the question, tool calls, answer, and cost/latency.

```python
def record(
    agent_name: str,
    problem: str,
    problem_context: dict,
    answer: dict,
    tokens_used: int,
    latency_ms: int,
    answer_text: str | None = None,
    tool_calls: list[dict] | None = None,
    category: str | None = None,
    source: str = "agent",
    *,
    context: dict | None = None,
) -> int
```

Example:

```python
kelt.atomic.solutions.record(
    agent_name="code-reviewer",
    problem="Review PR #123 for security issues",
    problem_context={"repo": "monorepo", "pr": 123, "diff_lines": 480},
    answer={"verdict": "approved", "findings": []},
    answer_text="No security issues found. Approved.",
    tool_calls=[{"name": "grep", "args": {"pattern": "SECRET_"}}],
    tokens_used=1500,
    latency_ms=2340,
    category="security-review",
)
```

Query methods:

```python
list_by_agent(agent_name, limit=100, active_only=True)  → list[Fact]
list_by_category(category, limit=100, active_only=True) → list[Fact]
get_agent_names()                                       → list[str]
get_stats(agent_name=None)                              → dict[str, int | float]
search(query, limit=50, active_only=True)               → list[Fact]  # ILIKE over problem
```

`get_stats` returns `{count, total_tokens, avg_latency_ms}` — the input for per-agent cost
dashboards.

## Relationships — `kelt.atomic.relationships`

Typed edges between two facts. Not a full graph store — just enough to record "this
assertion contradicts that one" or "this solution derives from this problem statement". For
richer graph structure, use [`kelt.kg`](knowledge-graph.md).

```python
def link(
    source_id: int,
    target_id: int,
    rel_type: RelType,
    confidence: float | None = 1.0,
    extra: dict | None = None,
) -> int    # edge id

def unlink(source_id, target_id, rel_type=None) → int  # rows deleted

def get_related(
    fact_id: int,
    rel_type: RelType | None = None,
    direction: Literal["outgoing", "incoming", "both"] = "both",
) -> list[FactRelationship]

def find_contradictions(fact_id: int | None = None, limit=1000) → list[FactRelationship]
def get_chain(fact_id, rel_type=RelType.DERIVED_FROM, max_depth=5, max_results=100) → list[FactRelationship]
def count(rel_type=None) → int
```

Example:

```python
from llm_kelt.memory.atomic.models import RelType

a1 = kelt.atomic.assertions.add("The service returns 500 on empty POST bodies")
a2 = kelt.atomic.assertions.add("The service returns 400 on empty POST bodies")

kelt.atomic.relationships.link(a1, a2, RelType.CONTRADICTS)

for c in kelt.atomic.relationships.find_contradictions():
    print(c.source_fact_id, "↯", c.target_fact_id)
```

`link()` raises `ConflictError` on a duplicate edge. `get_chain()` walks the graph via a
Postgres recursive CTE (14+).

## Embeddings — `kelt.atomic.embeddings`

Vector embeddings for facts. This is a thin adapter over `EmbeddingStoreClient` that
defaults `entity_type` to `"atomic.fact"` and returns `Fact` objects from `search_similar`.

Signatures and end-to-end use → [context and RAG](context-and-rag.md).

Quick shape:

```python
kelt.atomic.embeddings.set_embedding(fact_id, [0.12, -0.03, ...], model="text-embedding-3-small")

scored = kelt.atomic.embeddings.search_similar(
    query=query_vector,
    model="text-embedding-3-small",
    top_k=5,
    min_similarity=0.3,
    categories=["style"],
)
for sf in scored:
    print(sf.score, sf.entity.content)
```

## Bulk read across all types

```python
counts = kelt.atomic.get_stats()
# → {"assertions": 42, "feedback": 118, "preferences": 8, ...}
```

Combined with `Client.get_stats()`:

```python
stats = kelt.get_stats()
# → {"context_key": "…", "content": 87, "atomic": {…}}
```

## Cleanup

Deactivate is the safe default — the row stays for audit and can be re-activated:

```python
kelt.atomic.assertions.deactivate(fact_id)
```

Hard delete removes the fact and cascades to details, embeddings, and relationships:

```python
result = kelt.atomic.assertions.delete([fid1, fid2, fid3])
print(result.deleted_count)
```

Atomic memory has no bulk-by-scope delete — `context_key` is flat, not hierarchical. The KG
side has `kg.entities.delete_in_scope(scope_key)` for its own hierarchical entity scopes;
see [knowledge graph](knowledge-graph.md).
