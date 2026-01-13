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

---

## Adaptation Layer

The adaptation layer unifies all methods for customizing LLM behavior under a common interface.
Methods range from no-weight-change (context injection) to weight-change (LoRA, DPO).

### Adaptation Strategy Interface

```python
class AdaptationStrategy(Protocol):
    """Interface for all adaptation methods."""

    @property
    def name(self) -> str:
        """Strategy identifier (e.g., 'memory', 'lora', 'rag')."""
        ...

    @property
    def changes_weights(self) -> bool:
        """Whether this strategy modifies model weights."""
        ...

    async def prepare(self, profile_id: int) -> None:
        """
        Prepare strategy for inference.

        For memory: load facts from database
        For LoRA: load adapter weights
        For RAG: initialize retriever
        """
        ...

    async def adapt_request(
        self,
        messages: list[Message],
        system: str | None,
    ) -> AdaptedRequest:
        """
        Transform a request using this strategy.

        Returns modified messages/system prompt, or signals
        that model weights should be swapped.
        """
        ...
```

### Strategy Implementations

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        AdaptationStrategy                                │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
          ┌─────────────────────────┼─────────────────────────┐
          │                         │                         │
          ▼                         ▼                         ▼
┌───────────────────┐   ┌───────────────────┐   ┌───────────────────┐
│  MemoryStrategy   │   │    RAGStrategy    │   │   LoRAStrategy    │
│  (context inject) │   │ (retrieval+inject)│   │  (adapter swap)   │
├───────────────────┤   ├───────────────────┤   ├───────────────────┤
│ changes_weights:  │   │ changes_weights:  │   │ changes_weights:  │
│     False         │   │     False         │   │     True          │
├───────────────────┤   ├───────────────────┤   ├───────────────────┤
│ - Load facts      │   │ - Embed query     │   │ - Load adapter    │
│ - Format prompt   │   │ - Vector search   │   │ - Signal backend  │
│ - Inject context  │   │ - Inject results  │   │ - Swap weights    │
└───────────────────┘   └───────────────────┘   └───────────────────┘
```

### MemoryStrategy (Implemented)

Current fact injection approach - loads all active facts and injects into system prompt.

```python
class MemoryStrategy(AdaptationStrategy):
    name = "memory"
    changes_weights = False

    def __init__(self, facts_client: FactsClient):
        self._facts = facts_client

    async def prepare(self, profile_id: int) -> None:
        self._active_facts = self._facts.list_active()

    async def adapt_request(
        self,
        messages: list[Message],
        system: str | None,
    ) -> AdaptedRequest:
        facts_prompt = self._format_facts(self._active_facts)
        enhanced_system = f"{facts_prompt}\n\n{system or ''}"
        return AdaptedRequest(messages=messages, system=enhanced_system)
```

### RAGStrategy (Planned)

Semantic retrieval of relevant context based on query.

```python
class RAGStrategy(AdaptationStrategy):
    name = "rag"
    changes_weights = False

    def __init__(self, retriever: EmbeddingRetriever):
        self._retriever = retriever

    async def adapt_request(
        self,
        messages: list[Message],
        system: str | None,
    ) -> AdaptedRequest:
        query = self._extract_query(messages)
        relevant_docs = await self._retriever.search(query, top_k=10)
        context = self._format_context(relevant_docs)
        enhanced_system = f"{context}\n\n{system or ''}"
        return AdaptedRequest(messages=messages, system=enhanced_system)
```

### LoRAStrategy (Planned)

Load profile-specific LoRA adapter for weight-modified inference.

```python
class LoRAStrategy(AdaptationStrategy):
    name = "lora"
    changes_weights = True

    def __init__(self, adapter_registry: AdapterRegistry):
        self._registry = adapter_registry

    async def prepare(self, profile_id: int) -> None:
        self._adapter = await self._registry.get_adapter(profile_id)

    async def adapt_request(
        self,
        messages: list[Message],
        system: str | None,
    ) -> AdaptedRequest:
        return AdaptedRequest(
            messages=messages,
            system=system,
            adapter_path=self._adapter.path if self._adapter else None,
        )
```

### Composite Strategy

Combine multiple strategies (e.g., LoRA + Memory):

```python
class CompositeStrategy(AdaptationStrategy):
    """Apply multiple strategies in sequence."""

    def __init__(self, strategies: list[AdaptationStrategy]):
        self._strategies = strategies

    async def adapt_request(
        self,
        messages: list[Message],
        system: str | None,
    ) -> AdaptedRequest:
        request = AdaptedRequest(messages=messages, system=system)
        for strategy in self._strategies:
            request = await strategy.adapt_request(
                request.messages, request.system
            )
        return request
```

---

## Training Pipeline

Data flows from collection to trained artifacts.

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  COLLECTION │     │   EXPORT    │     │  TRAINING   │     │   DEPLOY    │
│             │     │             │     │             │     │             │
│ - feedback  │ --> │ - JSONL     │ --> │ - LoRA      │ --> │ - Registry  │
│ - prefs     │     │ - HF format │     │ - DPO       │     │ - Hot-swap  │
│ - content   │     │             │     │ - Classify  │     │             │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
```

### Export Formats

```python
class ExportFormat(Enum):
    JSONL = "jsonl"              # Generic line-delimited JSON
    ALPACA = "alpaca"            # instruction/input/output format
    SHAREGPT = "sharegpt"        # conversations format
    DPO_PAIRS = "dpo_pairs"      # chosen/rejected format
```

### Training Jobs

```python
@dataclass
class TrainingJob:
    id: str
    profile_id: int
    method: str                   # "lora", "dpo", "classifier"
    status: str                   # "pending", "running", "completed", "failed"
    config: TrainingConfig
    input_path: str               # Path to exported data
    output_path: str | None       # Path to trained artifact
    created_at: datetime
    completed_at: datetime | None
    metrics: dict | None          # Training metrics
```

### Adapter Registry

Manages trained adapters and their deployment.

```python
class AdapterRegistry:
    """Registry of trained adapters by profile."""

    async def register(
        self,
        profile_id: int,
        adapter_path: str,
        metadata: AdapterMetadata,
    ) -> str:
        """Register a new adapter, return adapter_id."""
        ...

    async def get_adapter(
        self,
        profile_id: int,
        version: str = "latest",
    ) -> Adapter | None:
        """Get adapter for profile."""
        ...

    async def list_adapters(
        self,
        profile_id: int,
    ) -> list[Adapter]:
        """List all adapter versions for profile."""
        ...

    async def set_active(
        self,
        profile_id: int,
        adapter_id: str,
    ) -> None:
        """Set active adapter for profile."""
        ...
```

### Adapter Storage

```sql
CREATE TABLE adapters (
    id VARCHAR(36) PRIMARY KEY,
    profile_id BIGINT NOT NULL REFERENCES profiles(id),
    method VARCHAR(50) NOT NULL,        -- 'lora', 'dpo', etc.
    version INT NOT NULL,
    path TEXT NOT NULL,                  -- Storage path
    base_model VARCHAR(100) NOT NULL,
    config JSONB,
    metrics JSONB,
    is_active BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE(profile_id, method, version)
);

CREATE INDEX idx_adapters_profile ON adapters(profile_id);
CREATE INDEX idx_adapters_active ON adapters(profile_id, is_active) WHERE is_active;
```

---

## Integration: Serving with Adaptation

Updated serving flow with pluggable adaptation:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Client Request                                 │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          Serving Proxy                                   │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    AdaptationOrchestrator                        │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │   │
│  │  │   Memory    │  │    RAG      │  │    LoRA     │              │   │
│  │  │  Strategy   │  │  Strategy   │  │  Strategy   │              │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘              │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           Backend LLM                                    │
│                    (with optional adapter loaded)                        │
└─────────────────────────────────────────────────────────────────────────┘
```

### AdaptationOrchestrator

Coordinates multiple strategies:

```python
class AdaptationOrchestrator:
    def __init__(
        self,
        strategies: list[AdaptationStrategy],
        profile_id: int,
    ):
        self._strategies = strategies
        self._profile_id = profile_id

    async def prepare(self) -> None:
        """Prepare all strategies for this profile."""
        for strategy in self._strategies:
            await strategy.prepare(self._profile_id)

    async def adapt(
        self,
        messages: list[Message],
        system: str | None,
    ) -> AdaptedRequest:
        """Apply all strategies to the request."""
        request = AdaptedRequest(messages=messages, system=system)
        for strategy in self._strategies:
            request = await strategy.adapt_request(
                request.messages, request.system
            )
        return request
```

### Server Configuration

```yaml
serving:
  profile_id: 1
  adaptation:
    strategies:
      - type: memory
        enabled: true
      - type: rag
        enabled: false
        config:
          top_k: 10
          min_similarity: 0.7
      - type: lora
        enabled: false
        config:
          auto_load: true
```
