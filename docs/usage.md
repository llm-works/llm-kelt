# Usage Guide

How to use the llm-kelt.

---

## Setup

```python
from appinfra.config import Config
from appinfra.log import LogConfig, LoggerFactory
from llm_kelt import ClientContext, ClientFactory

# Load configuration
config = Config("etc/llm-kelt.yaml")
lg = LoggerFactory.create_root(LogConfig.from_params(level="info"))

# Create client via factory
factory = ClientFactory(lg)
context = ClientContext(context_key="default")
client = factory.create_from_config(context=context, config=config)

# Check connection
client.health_check()
```

---

## Multi-Schema Operations

Use `with_schema()` for per-operation schema selection. This is useful when agents need to
write to different schemas based on runtime context (e.g., training manifests specifying
target schemas).

```python
# Schema-agnostic client (no schema_name in context)
context = ClientContext(context_key="my-agent")
client = factory.create_from_config(context=context, config=config)

# Schema specified at operation time
client.with_schema("hn_exp").atomic.solutions.record(
    agent_name="reviewer",
    problem="How to optimize database queries?",
    solution="Use indexes and query plans...",
)

client.with_schema("playground").atomic.assertions.add(
    "User prefers concise responses",
    category="preferences",
)

# Schema from training manifest
schema = manifest.source.schema_name if manifest.source else "default"
client.with_schema(schema).atomic.facts.add(...)
```

**Key behaviors:**

- **Lazy initialization** - Schema and tables created on first `.atomic` access (when `schema_mode=SchemaMode.ENSURE`)
- **Lightweight** - `ScopedClient` shares resources (embedder, logger) with parent
- **Independent scopes** - Each `with_schema()` call returns a fresh `ScopedClient`
- **Context key preserved** - All scoped operations use the parent's `context_key`

---

## Conversations

Stateful multi-turn dialogue with token tracking, compaction, and persistence.

### Basic Usage

```python
from llm_kelt.conversation import Conversation, Config, Role

# Create conversation with token limit (lg is a Logger instance)
conv = Conversation(lg, config=Config(max_tokens=32000))
conv.add("You are a helpful assistant.", Role.SYSTEM)
conv.add("What is a Python decorator?")
conv.add("A decorator wraps another function to extend its behavior.", Role.ASSISTANT)

# Check state
conv.message_count  # 3
conv.token_count  # estimated tokens
conv.usage_ratio  # fraction of limit used
conv.needs_compaction()  # True when usage exceeds threshold
```

### Tool Call Messages

```python
conv.add(
    "",
    Role.ASSISTANT,
    tool_calls=[
        {"id": "tc_1", "name": "list_files", "arguments": {"path": "."}},
    ],
)
conv.add("main.py\nutils.py", Role.TOOL, tool_call_id="tc_1")
```

### Compaction

When conversations approach token limits, compactors reduce size while preserving context.
**Hard limit guarantee**: After compaction, `token_count` must be <= `max_tokens`.
If compaction cannot achieve this, `ContextOverflowError` is raised.

```python
from llm_kelt.conversation import SlidingWindowCompactor, ContextOverflowError

# Manual compaction
compactor = SlidingWindowCompactor()
if conv.needs_compaction():
    compactor.compact(conv)  # drops oldest messages, keeps system + recent

# Auto-compaction via injected compactor
conv = Conversation(
    lg,
    config=Config(max_tokens=32000, compact_threshold=0.8, min_recent_messages=4),
    compactor=SlidingWindowCompactor(),
)
# Compaction fires automatically when threshold is exceeded on add()
# Raises ContextOverflowError if compaction cannot reduce below max_tokens
```

For accurate token counting (instead of char/4 heuristic), provide a tokenizer:

```python
import tiktoken

enc = tiktoken.encoding_for_model("gpt-4")
conv = Conversation(
    lg,
    config=Config(max_tokens=24000, tokenizer=lambda text: len(enc.encode(text))),
    compactor=SlidingWindowCompactor(),
)
```

For better context retention, use the summarizing compactor (requires an LLM client):

```python
from llm_kelt.conversation import SummarizingCompactor

compactor = SummarizingCompactor(client=llm_client, model="qwen2.5-7b")
# Summarizes old messages via LLM before discarding them
```

### Session Persistence

Save and restore conversations using file or database storage.

```python
from llm_kelt.conversation.storage import FileSessionStorage

storage = FileSessionStorage(lg, "~/.my-agent/sessions")

# Save
storage.save("session-123", conv, metadata={"model": "qwen2.5-7b"})

# List sessions
for s in storage.list():
    print(f"{s.session_id}  msgs={s.message_count}  tokens={s.token_count}")

# Load
loaded = storage.load("session-123")
# loaded.messages, loaded.metadata, loaded.config, loaded.token_count

# Delete
storage.delete("session-123")
```

### With ContextQuery

The conversation layer integrates with `ContextQuery` for multi-turn RAG:

```python
from llm_kelt.conversation import Conversation, Config
from llm_kelt.inference.query import ContextQuery

conv = Conversation(lg, config=Config(max_tokens=4000))
query = ContextQuery(client=llm_client, context_builder=builder)

response = await query.ask("What are Python generators?", conversation=conv)
response = await query.ask("How do they differ from lists?", conversation=conv)
# conv tracks the full multi-turn history automatically
```

---

## Recording Feedback

Explicit signals about content quality.

```python
# Positive feedback
client.feedback.record(
    content_text="The response explaining Docker networking...",
    signal="positive",
    strength=0.9,  # 0.0-1.0
    tags=["docker", "networking"],
    comment="Clear and concise",
)

# Negative feedback
client.feedback.record(
    content_text="The rambling explanation of...",
    signal="negative",
    strength=0.8,
)

# Dismiss (not relevant)
client.feedback.record(
    content_text="...",
    signal="dismiss",
)
```

---

## Recording Preferences

Comparisons for DPO training.

```python
client.preferences.record(
    context="Explain kubernetes pods",
    chosen="Pods are the smallest deployable units...",
    rejected="Kubernetes is a container orchestration platform...",
    margin=0.7,  # How much better (0.0-1.0)
    domain="infrastructure",
)
```

---

## Storing Content

For RAG and reference.

```python
# Create content
content_id = client.content.create(
    content_text="Full article text...",
    source="arxiv",
    external_id="2401.12345",
    url="https://arxiv.org/abs/2401.12345",
    title="Attention Is All You Need",
)

# Get or create (deduplicates by hash)
content_id, created = client.content.get_or_create(
    content_text="Same text...",
    source="manual",
)
```

---

## Tracking Predictions

For calibration analysis.

```python
# Record prediction
pred_id = client.predictions.record(
    hypothesis="The refactor will take less than 3 days",
    confidence=0.75,
    resolution_date="2025-01-20",
    domain="engineering",
    tags=["estimates", "refactoring"],
)

# Later: resolve when outcome known
client.predictions.resolve(
    prediction_id=pred_id,
    outcome="incorrect",  # correct, incorrect, partial, cancelled
    actual="Took 5 days due to unforeseen API changes",
)

# Get calibration data
data = client.predictions.get_calibration_data()
# [(0.75, False), (0.60, True), ...]
```

---

## Managing Directives

Standing instructions for the AI.

```python
# Add directive
client.directives.record(
    text="Always provide code examples in Python",
    directive_type="standing",  # standing, one-time, rule
)

# Add expiring directive
from datetime import datetime, timedelta

client.directives.record(
    text="Focus on kubernetes this week",
    directive_type="one-time",
    expires_at=datetime.now() + timedelta(days=7),
)

# Get active directives
active = client.directives.list_active()

# Pause/complete directive
client.directives.set_status(directive_id, "paused")
client.directives.set_status(directive_id, "completed")
```

---

## Exporting Training Data

```python
from llm_kelt.core.export import (
    export_feedback,
    export_preferences,
    export_predictions,
    load_jsonl,
)

# Export feedback for classifier training
count = export_feedback(
    client._db,
    "feedback.jsonl",
    signals=["positive", "negative"],
    since=datetime(2025, 1, 1),
)
print(f"Exported {count} records")

# Export preferences for DPO
export_preferences(
    client._db,
    "preferences.jsonl",
    domain="infrastructure",
)

# Export predictions for calibration
export_predictions(
    client._db,
    "predictions.jsonl",
    status="resolved",
)

# Load back
records = load_jsonl("feedback.jsonl")
```

---

## Querying Data

```python
# Feedback
feedback_list = client.feedback.list_by_signal("positive", limit=50)
counts = client.feedback.count_by_signal()
# {"positive": 42, "negative": 15, "dismiss": 8}

# Preferences
pairs = client.preferences.list_by_domain("infrastructure")
domains = client.preferences.list_domains()

# Predictions
pending = client.predictions.list_pending()
due = client.predictions.list_due()  # Past resolution date
resolved = client.predictions.list_resolved(outcome="correct")

# Directives
active = client.directives.list_active()
by_type = client.directives.list_by_type("rule")

# Generic (all clients)
item = client.feedback.get(id=123)
items = client.feedback.list(limit=100, offset=0)
count = client.feedback.count()
deleted = client.feedback.delete(id=123)
```

---

## Stats

```python
stats = client.get_stats()
# {
#     "feedback": 65,
#     "preferences": 23,
#     "content": 150,
#     "interactions": 0,
#     "predictions": 12,
#     "directives": 5,
# }
```
