# Architecture

## System Overview

```
+------------------------------------------------------------------+
|                         LearnClient                               |
|              (Profile-scoped API entry point)                     |
+------------------------------------------------------------------+
         |                    |                    |
         v                    v                    v
+-----------------+  +-----------------+  +-----------------+
|   COLLECTION    |  |    INFERENCE    |  |     SERVING     |
| Data Collection |  |  LLM + Context  |  |   Proxy Server  |
+-----------------+  +-----------------+  +-----------------+
| - facts         |  | - LLMClient     |  | - Fact injection|
| - feedback      |  | - ContextBuilder|  | - Backend proxy |
| - preferences   |  | - ContextQuery  |  | - OpenAI compat |
| - predictions   |  | - Backends      |  |                 |
| - interactions  |  |   (OpenAI, Anth)|  |                 |
| - content       |  |                 |  |                 |
+-----------------+  +-----------------+  +-----------------+
         |                    |                    |
         +--------------------+--------------------+
                              |
                    +-----------------+
                    |      CORE       |
                    |  Database/Models|
                    +-----------------+
                    | - Database      |
                    | - Models (ORM)  |
                    | - Export (JSONL)|
                    | - Exceptions    |
                    +-----------------+
```

## Component Responsibilities

### LearnClient

Main entry point, scoped to a profile. Provides unified access to all collection APIs.

```python
from llm_learn import LearnClient

learn = LearnClient(profile_id=1)
learn.facts.add("Prefers concise output", category="preferences")
learn.feedback.record(content_text="...", signal="positive")
learn.preferences.record(context="...", chosen="...", rejected="...")
```

### Collection Module

Profile-scoped clients for data collection:

| Client | Purpose | Key Methods |
|--------|---------|-------------|
| `FactsClient` | Store facts for context injection | `add()`, `list_active()`, `deactivate()` |
| `FeedbackClient` | Positive/negative signals | `record()`, `list()`, `export_jsonl()` |
| `PreferencesClient` | Chosen/rejected pairs for DPO | `record()`, `list()`, `export_jsonl()` |
| `PredictionsClient` | Hypothesis tracking | `record()`, `resolve()`, `get_calibration()` |
| `InteractionsClient` | Engagement metrics | `record()`, `list()` |
| `ContentClient` | Store content with embeddings | `store()`, `get()`, `search_similar()` |

### Inference Module

LLM interaction with context injection:

| Component | Purpose |
|-----------|---------|
| `LLMClient` | Unified client for multiple backends (OpenAI-compatible, Anthropic) |
| `ContextBuilder` | Builds system prompts with injected facts |
| `ContextQuery` | High-level interface for context-aware queries |

```python
from llm_learn.inference import LLMClient, ContextBuilder

client = LLMClient.from_config(config["llm"])
context = ContextBuilder(learn.facts)
system = context.build_system_prompt("You are a helpful assistant.")
response = await client.chat(messages, system=system)
```

### Serving Module

OpenAI-compatible proxy server that injects facts into requests:

```python
from llm_learn.serving import create_server

server = create_server(
    llm_config=config["llm"],
    database=db,
    profile_id=1,
)
server.start()  # Serves on :8001
```

Intercepts `/v1/chat/completions`, injects facts, forwards to backend.

### Core Module

Shared infrastructure:

| Component | Purpose |
|-----------|---------|
| `Database` | PostgreSQL connection, migrations |
| `models.py` | SQLAlchemy ORM models |
| `export/jsonl.py` | Training data export |
| `exceptions.py` | Framework exceptions |

---

## Data Flow

### Fact Injection Flow

```
Client Request
       |
       v
+-------------+     get facts     +-------------+
|   SERVING   | ----------------> |    LEARN    |
|   (Proxy)   |                   |   (Facts)   |
+-------------+                   +-------------+
       |
       | inject into system prompt
       v
+-------------+
|   BACKEND   |  (OpenAI, Anthropic, local)
|     LLM     |
+-------------+
       |
       v
   Response
```

### Feedback Collection Flow

```
User provides feedback
       |
       v
+-------------+
|    LEARN    |
|  (Feedback) |
+-------------+
       |
       v
  PostgreSQL
  (feedback table)
       |
       v
  JSONL Export --> Training Pipeline
```

### Training Data Export

```
+-------------+
|    LEARN    |
+-------------+
       |
       | learn.feedback.export_jsonl("feedback.jsonl")
       | learn.preferences.export_jsonl("prefs.jsonl")
       v
+-------------+
|    JSONL    |  --> External training (LoRA, classifiers)
|    Files    |
+-------------+
```

---

## Interface Definitions

### LearnClient API

```python
class LearnClient:
    def __init__(self, profile_id: int, database: Database | None = None): ...

    @property
    def facts(self) -> FactsClient: ...
    @property
    def feedback(self) -> FeedbackClient: ...
    @property
    def preferences(self) -> PreferencesClient: ...
    @property
    def predictions(self) -> PredictionsClient: ...
    @property
    def interactions(self) -> InteractionsClient: ...
    @property
    def content(self) -> ContentClient: ...

    def get_stats(self) -> dict: ...
    def health_check(self) -> dict: ...
```

### FactsClient API

```python
class FactsClient:
    def add(
        self,
        content: str,
        category: str | None = None,
        source: str = "manual",
        confidence: float = 1.0,
    ) -> Fact: ...

    def list_active(
        self,
        category: str | None = None,
        min_confidence: float = 0.0,
        limit: int = 100,
    ) -> list[Fact]: ...

    def deactivate(self, fact_id: int) -> None: ...
```

### FeedbackClient API

```python
class FeedbackClient:
    def record(
        self,
        content_text: str,
        signal: Literal["positive", "negative", "dismiss"],
        context: dict | None = None,
    ) -> Feedback: ...

    def export_jsonl(self, path: str) -> int: ...
```

### LLMClient API

```python
class LLMClient:
    @classmethod
    def from_config(cls, config: dict) -> LLMClient: ...

    async def chat(
        self,
        messages: list[Message],
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> ChatResponse: ...

    async def chat_full(
        self,
        messages: list[Message],
        **kwargs,
    ) -> ChatResponse: ...
```

---

## Storage

### Database Schema

See [Data Model](05-data-model.md) for full schema.

Key tables:
- `profiles` - User profiles
- `facts` - Stored facts for context injection
- `feedback` - Positive/negative signals
- `preference_pairs` - Chosen/rejected for DPO
- `predictions` - Hypothesis tracking
- `content` - Stored content with embeddings

### File Exports

```
exports/
+-- feedback_2024_01.jsonl      # Training data
+-- preferences_2024_01.jsonl   # DPO pairs
```

---

## Deployment

### Single Process

```python
from llm_learn import LearnClient
from llm_learn.serving import create_server

learn = LearnClient(profile_id=1)
server = create_server(llm_config=config, database=learn.database, profile_id=1)
server.start()
```

### With External LLM

```
+------------------+     +------------------+
|  llm-learn proxy |     |  LLM Backend     |
|      :8001       | --> |  (vLLM :8000)    |
+------------------+     +------------------+
         |
         v
+------------------+
|   PostgreSQL     |
+------------------+
```

---

## Backend Support

### OpenAI-Compatible
- vLLM, LMStudio, Ollama, text-generation-inference
- Any server implementing `/v1/chat/completions`

### Anthropic
- Claude API via `anthropic` SDK

### Configuration

```yaml
llm:
  default: local
  backends:
    local:
      type: openai_compatible
      base_url: http://localhost:8000/v1
      model: qwen2.5-72b
    anthropic:
      type: anthropic
      model: claude-sonnet-4-20250514
```
