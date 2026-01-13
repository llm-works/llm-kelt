# Framework Strategy

Which approaches the framework supports, how responsibilities are divided between learn and agent.

## Architectural Boundary

**learn** and **agent** have distinct responsibilities:

| Layer | Responsibility | Examples |
|-------|---------------|----------|
| **learn** | Data primitives + pipeline framework | Facts CRUD, feedback storage, vector search, pipeline executor |
| **agent** | Orchestration intelligence | Routing decisions, quality evaluation, model selection, retry logic |

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              agent                                       │
│                                                                          │
│   Implements learn's protocols:                                          │
│   - ContextSelector (which facts for this query?)                        │
│   - QualityEvaluator (is response good enough?)                          │
│   - ModelRouter (small or large model?)                                  │
│   - RetryStrategy (how to handle failures?)                              │
│                                                                          │
│   Uses learn's primitives + pipeline framework                           │
└──────────────────────────────────────────────────────────────────────────┘
                                   │
                        implements │ protocols
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                              learn                                       │
│                                                                          │
│   Provides:                                                              │
│   - Data primitives (facts, feedback, preferences, content)              │
│   - Vector search (similarity retrieval)                                 │
│   - Pipeline framework (executor calls agent-provided hooks)             │
│   - Protocol definitions (what agent must implement)                     │
│   - Export functionality (training data)                                 │
│                                                                          │
│   Does NOT provide:                                                      │
│   - Routing logic (agent decides)                                        │
│   - Quality evaluation (agent decides)                                   │
│   - Model selection (agent decides)                                      │
└─────────────────────────────────────────────────────────────────────────┘
```

**Key insight:** learn defines *what* the pipeline does (structure, data flow, hooks). Agent defines
*how* decisions are made (implementations of the hooks).

---

## Approach Selection

From the [learning methods](02-learning-methods.md), the framework supports:

| Approach | Use Case | Owner |
|----------|----------|-------|
| **RAG** | Semantic context retrieval | learn (primitives) + agent (selection logic) |
| **Memory** | Fact injection | learn (storage) + agent (routing) |
| **Classifiers** | Fast scoring/filtering | learn (data export) + agent (model calls) |
| **Embeddings** | Similarity search | learn (vector storage + search) |
| **LoRA + DPO** | Behavior customization | learn (training data export) |

## What learn Provides

### Data Primitives

```python
# learn owns CRUD and retrieval
class FactsClient:
    def add(content, category, source, confidence) -> int
    def list_active(category, min_confidence, limit) -> list[Fact]
    def search(query, active_only, limit) -> list[Fact]

class ContentClient:
    def store(content_text, source, ...) -> int
    def search_similar(query, top_k, min_similarity) -> list[ScoredContent]

class FeedbackClient:
    def record(signal, content_text, context) -> int
    def export_jsonl(path) -> int
```

### Pipeline Framework

learn provides the pipeline executor that calls agent-provided hooks:

```python
# learn owns the pipeline structure
class AdaptationPipeline:
    def __init__(
        self,
        learn_client: LearnClient,
        # Agent provides these implementations
        context_selector: ContextSelector,
        quality_evaluator: QualityEvaluator,
        model_router: ModelRouter,
        retry_strategy: RetryStrategy,
        ...
    ): ...

    async def process(query, messages, system) -> PipelineResult:
        # learn controls the flow, agent provides the intelligence
        ...
```

### Protocol Definitions

learn defines what agent must implement:

```python
# learn defines the contracts
class ContextSelector(Protocol):
    async def select(query, candidates, max_facts) -> SelectionResult: ...

class QualityEvaluator(Protocol):
    async def evaluate(query, response, context) -> EvaluationResult: ...

class ModelRouter(Protocol):
    async def route(query, context_confidence) -> RoutingDecision: ...

class RetryStrategy(Protocol):
    async def should_retry(attempt, quality_score, error) -> RetryDecision: ...
```

See [Context Routing](06-context-routing.md) and [Advanced Methods](07-advanced-methods.md) for full
protocol definitions.

## What learn Does NOT Provide

| Concern | Why Not | Owner |
|---------|---------|-------|
| **Routing logic** | Requires LLM calls, domain knowledge | agent |
| **Quality evaluation** | Requires LLM judge or domain rules | agent |
| **Model selection** | Requires budget/capability awareness | agent |
| **Retry logic** | Requires strategy decisions | agent |

## What the Framework Does NOT Support

| Approach | Reason |
|----------|--------|
| **Full fine-tuning** | Too expensive, LoRA is sufficient |
| **RLHF** | Overkill, DPO achieves similar results more simply |
| **Training from scratch** | Uses existing base models |
| **Continued pre-training** | LoRA + RAG achieves similar results at fraction of cost |
| **Prompt tuning** | Not available via API, limited value |
| **Reward modeling** | Only needed for RLHF, which we're not doing |

---

## Implementation Layers

### Layer 1: Data Primitives (learn)

**Goal:** Store and retrieve personalization data.

- Facts storage with categories and confidence
- Feedback collection (positive/negative/dismiss)
- Preference pairs for DPO training
- Content deduplication with embeddings
- Vector similarity search

**No intelligence here** - just CRUD and retrieval.

### Layer 2: Pipeline Framework (learn)

**Goal:** Define the adaptation flow structure.

- Pipeline executor with configurable steps
- Protocol definitions for agent hooks
- Data types for pipeline state
- Result recording back to learn

**Orchestrates the flow, delegates decisions to agent.**

### Layer 3: Orchestration Intelligence (agent)

**Goal:** Make smart decisions within the pipeline.

- Context selection (rules, embeddings, LLM-based)
- Quality evaluation (rule-based, LLM judge)
- Model routing (small for filtering, large for reasoning)
- Retry strategies (escalate model, expand context)

**Implements learn's protocols with actual intelligence.**

---

## Integration Example

```python
# In agent - assembles pipeline with its implementations

from llm_learn import LearnClient
from llm_learn.adaptation import AdaptationPipeline, PipelineConfig

# Agent's implementations of learn's protocols
from agent.adaptation.selectors import HybridSelector
from agent.adaptation.evaluators import LLMJudgeEvaluator
from agent.adaptation.routers import ConfidenceBasedRouter
from agent.adaptation.retry import ExponentialBackoffRetry

class OrchestrationAgent:
    def setup(self):
        # Learn client for data primitives
        self.learn = LearnClient(profile_id=self.config.profile_id)

        # Agent provides implementations
        selector = HybridSelector(self.small_model)
        evaluator = LLMJudgeEvaluator(self.small_model)
        router = ConfidenceBasedRouter("small", "large", threshold=0.7)
        retry = ExponentialBackoffRetry(max_attempts=3)

        # Learn's pipeline with agent's implementations
        self.pipeline = AdaptationPipeline(
            learn_client=self.learn,
            context_selector=selector,
            quality_evaluator=evaluator,
            model_router=router,
            retry_strategy=retry,
            models={"small": self.small_model, "large": self.large_model},
            config=PipelineConfig(max_attempts=3),
        )
```

---

## Trade-offs

### Simplicity over Power
- LoRA over full fine-tune
- DPO/KTO over RLHF (no reward model needed)
- Protocol-based hooks over hardcoded logic

### Portability over Lock-in
- Store everything in PostgreSQL (portable)
- JSONL exports for training (tool-agnostic)
- Support both API and local models

### Separation of Concerns
- learn = data + framework (stable, tested)
- agent = intelligence (flexible, evolving)
- Clean protocol boundary between them

---

## Resource Requirements

| Layer | GPU | Storage |
|-------|-----|---------|
| Layer 1 (Data) | None | PostgreSQL + pgvector |
| Layer 2 (Pipeline) | None | Same |
| Layer 3 (Agent) | Optional for local models | + model weights |

For agent with local models:
- Small model (filtering): 8GB GPU
- Large model (reasoning): 24-96GB GPU or API

---

## Success Metrics

| Layer | Metric | Target |
|-------|--------|--------|
| Layer 1 | Data integrity, query latency | <100ms retrieval |
| Layer 2 | Pipeline completion rate | >95% |
| Layer 3 | Quality score improvement | >20% vs no adaptation |
