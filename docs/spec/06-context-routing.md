# Context Routing Architecture

## Problem

As the fact store grows, dumping all facts into the system prompt becomes impractical:
- Token limits exceeded
- Irrelevant context dilutes important information
- Response quality degrades

We need intelligent context selection: the right facts for each query.

## Approach: Hybrid Routing

Three-tier system that escalates complexity only when needed:

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Query                               │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Tier 1: Rule-Based                            │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ • Category mapping (coding question → programming facts)  │  │
│  │ • Always-include facts (core preferences, identity)       │  │
│  │ • Keyword triggers (explicit mentions)                    │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                                │
                    (if insufficient or ambiguous)
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Tier 2: Embedding Retrieval                     │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ • Embed query                                              │  │
│  │ • Vector similarity search against fact embeddings        │  │
│  │ • Return top-k candidates                                  │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                                │
                    (if confidence low or ambiguous)
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Tier 3: Router LLM                            │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ • Small/fast model for routing decisions                  │  │
│  │ • Input: query + candidate facts from Tier 2              │  │
│  │ • Output: selected facts + relevance reasoning            │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Inference LLM                               │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ • Full model with selected context                        │  │
│  │ • System prompt with routed facts                         │  │
│  │ • User query                                               │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Components

### 1. Fact Store with Embeddings

Extend current `facts` table:

```sql
ALTER TABLE facts ADD COLUMN embedding vector(1536);
CREATE INDEX idx_facts_embedding ON facts USING ivfflat (embedding vector_cosine_ops);
```

Embedding generation:
- On fact creation/update
- Batch job for existing facts
- Consistent embedding model across query and facts

### 2. Rule Engine

Configuration-driven rules for deterministic routing:

```yaml
routing:
  always_include:
    - category: "identity"
    - category: "core_preferences"

  category_mapping:
    - triggers: ["code", "programming", "debug", "function"]
      categories: ["programming", "technical"]
    - triggers: ["schedule", "meeting", "calendar"]
      categories: ["scheduling", "preferences"]

  keyword_triggers:
    - pattern: "my name"
      categories: ["identity"]
```

### 3. Embedding Retriever

```python
class EmbeddingRetriever:
    async def retrieve(
        self,
        query: str,
        profile_id: int,
        top_k: int = 30,
        min_similarity: float = 0.7,
    ) -> list[ScoredFact]:
        """
        Retrieve facts by embedding similarity.
        Returns facts with similarity scores for downstream filtering.
        """
        query_embedding = await self.embed(query)
        return await self.vector_search(
            embedding=query_embedding,
            profile_id=profile_id,
            top_k=top_k,
            min_similarity=min_similarity,
        )
```

### 4. Router LLM

Dedicated model for context selection:

```python
class RouterLLM:
    SYSTEM_PROMPT = """
    You are a context selection assistant. Given a user query and candidate facts,
    select the facts most relevant to answering the query.

    Output JSON: {"selected_ids": [1, 3, 7], "reasoning": "..."}

    Selection criteria:
    - Directly relevant to the query topic
    - Provides necessary background for accurate response
    - Respects user preferences that affect response style
    """

    async def select(
        self,
        query: str,
        candidates: list[Fact],
        max_facts: int = 10,
    ) -> RouterDecision:
        ...
```

### 5. Routing Cache

Cache routing decisions for repeated query patterns:

```python
class RoutingCache:
    async def get(self, query: str, profile_id: int) -> CachedRoute | None:
        """Return cached fact selection if fresh."""
        ...

    async def set(
        self,
        query: str,
        profile_id: int,
        fact_ids: list[int],
        tier_used: str,
    ):
        """Cache routing decision."""
        ...
```

Cache invalidation triggers:
- Fact creation/update/deletion
- Time-based expiry
- Profile configuration changes

## Routing Orchestrator

```python
class HybridRouter:
    async def route(self, query: str, profile_id: int) -> RoutingResult:
        # Check cache first
        cached = await self.cache.get(query, profile_id)
        if cached:
            return cached.to_result()

        # Tier 1: Rules
        rule_facts = await self.rule_engine.match(query, profile_id)
        always_facts = await self.rule_engine.always_include(profile_id)

        if self._rules_sufficient(query, rule_facts):
            return RoutingResult(
                facts=always_facts + rule_facts,
                tier="rules",
            )

        # Tier 2: Embeddings
        candidates = await self.retriever.retrieve(query, profile_id)

        if self._high_confidence(candidates):
            selected = self._select_top(candidates, k=10)
            return RoutingResult(
                facts=always_facts + selected,
                tier="embedding",
            )

        # Tier 3: Router LLM
        decision = await self.router_llm.select(query, candidates)

        return RoutingResult(
            facts=always_facts + decision.facts,
            tier="router_llm",
            reasoning=decision.reasoning,
        )

    def _high_confidence(self, candidates: list[ScoredFact]) -> bool:
        """
        Determine if embedding results are confident enough.

        High confidence when:
        - Top results have high similarity scores
        - Clear separation between relevant and irrelevant
        """
        ...
```

## Data Model Extensions

### RoutingResult

```python
@dataclass
class RoutingResult:
    facts: list[Fact]
    tier: str  # "rules" | "embedding" | "router_llm"
    reasoning: str | None = None
    candidates_considered: int = 0
```

### Routing Feedback

Track routing decisions for analysis and improvement:

```sql
CREATE TABLE routing_decisions (
    id BIGSERIAL PRIMARY KEY,
    profile_id BIGINT NOT NULL REFERENCES profiles(id),
    query_hash VARCHAR(64) NOT NULL,
    tier_used VARCHAR(20) NOT NULL,
    selected_fact_ids BIGINT[] NOT NULL,
    candidates_count INT,
    reasoning TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE routing_feedback (
    id BIGSERIAL PRIMARY KEY,
    routing_decision_id BIGINT NOT NULL REFERENCES routing_decisions(id),
    signal VARCHAR(20) NOT NULL,  -- 'positive', 'negative', 'missing_context'
    comment TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

## Integration with Proxy

Update the serving layer to use hybrid routing:

```python
class _RouteHandlers:
    def __init__(
        self,
        model_name: str,
        llm_client: LLMClient,
        router: HybridRouter,  # Replaces simple ContextBuilder
        streaming_client: OpenAIClient | None = None,
    ):
        self.router = router
        ...

    async def chat_completions(self, body: ChatCompletionRequest):
        # Route to get relevant facts
        routing_result = await self.router.route(
            query=self._extract_query(body.messages),
            profile_id=self.profile_id,
        )

        # Build system prompt from routed facts
        system_prompt = self._build_system_prompt(routing_result.facts)

        # Continue with inference...
```

## Open Questions

1. **Escalation thresholds**: How to tune confidence thresholds for tier escalation?
2. **Router model**: Which model balances speed and accuracy for routing?
3. **Fact grouping**: Should related facts be grouped/summarized?
4. **Multi-turn**: How does conversation history affect routing?
5. **Feedback collection**: How to gather implicit feedback on routing quality?
