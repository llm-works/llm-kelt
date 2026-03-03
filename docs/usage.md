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
schema = manifest.source.schema_name or "default"
client.with_schema(schema).atomic.facts.add(...)
```

**Key behaviors:**

- **Lazy initialization** - Schema and tables created on first `.atomic` access
- **Lightweight** - `ScopedClient` shares resources (embedder, logger) with parent
- **Independent scopes** - Each `with_schema()` call returns a fresh `ScopedClient`
- **Context key preserved** - All scoped operations use the parent's `context_key`

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
