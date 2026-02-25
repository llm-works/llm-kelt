# Context Routing

How kelt provides primitives for context selection, and the protocols agent implements.

## Problem

Dumping all facts into the system prompt doesn't scale:

```python
# Naive approach - doesn't work
system = base_prompt + "\n".join(all_facts)  # 500 facts = 50k tokens = slow, expensive, confused
```

We need intelligent selection: given a query, which facts are relevant?

## Architectural Split

| Layer | Responsibility |
|-------|---------------|
| **kelt** | Vector search primitives, candidate retrieval, protocol definitions |
| **agent** | Selection logic (rules, scoring, LLM-based selection) |

kelt provides the **what** (candidates, data types). Agent provides the **how** (selection logic).

---

## Kelt Provides: Primitives

### Vector Search

```python
# kelt/collection/content.py
class ContentClient:
    async def search_similar(
        self,
        query: str,
        top_k: int = 30,
        min_similarity: float = 0.5,
    ) -> list[ScoredContent]:
        """
        Retrieve content candidates by embedding similarity.

        Returns candidates sorted by similarity score.
        Agent decides final selection from these candidates.
        """
        query_embedding = await self._embedder.embed_async(query)
        return await self._vector_store.search(
            embedding=query_embedding,
            top_k=top_k,
            min_similarity=min_similarity,
        )
```

### Fact Retrieval

```python
# kelt/collection/facts.py
class FactsClient:
    async def list_by_category(
        self,
        category: str,
        include_inactive: bool = False,
    ) -> list[Fact]:
        """Get all facts in a category."""
        ...

    async def list_active(
        self,
        category: str | None = None,
        min_confidence: float = 0.0,
        limit: int = 100,
    ) -> list[Fact]:
        """Get active facts, optionally filtered."""
        ...
```

### Database Support

Extend facts table for embeddings:

```sql
ALTER TABLE facts ADD COLUMN embedding vector(1536);
CREATE INDEX idx_facts_embedding ON facts USING ivfflat (embedding vector_cosine_ops);
```

---

## Kelt Defines: Data Types

```python
# kelt/adaptation/types.py

@dataclass
class ScoredFact:
    """Fact with relevance score from vector search."""
    fact: Fact
    similarity: float  # 0-1, from embedding search


@dataclass
class SelectionResult:
    """Result of context selection (agent produces this)."""
    facts: list[Fact]
    confidence: float  # How confident in selection (0-1)
    tier: str  # "rules" | "embedding" | "llm" - which method was used
    reasoning: str | None = None
    candidates_considered: int = 0
```

---

## Kelt Defines: Protocols

Agent must implement these protocols to participate in the pipeline.

### ContextSelector Protocol

```python
# kelt/adaptation/protocols.py

from typing import Protocol, runtime_checkable

@runtime_checkable
class ContextSelector(Protocol):
    """
    Select relevant context for a query.

    Agent implements this with rules, embeddings, LLM, or hybrid approaches.
    Kelt's pipeline calls this during adaptation.
    """

    async def select(
        self,
        query: str,
        candidates: list[ScoredFact],
        max_facts: int = 10,
    ) -> SelectionResult:
        """
        Given candidates from vector search, select the most relevant.

        Args:
            query: The user's query
            candidates: Pre-filtered candidates from kelt's vector search
            max_facts: Maximum facts to return

        Returns:
            SelectionResult with selected facts, confidence, and method used
        """
        ...
```

### RoutingCache Protocol (Optional)

```python
@runtime_checkable
class RoutingCache(Protocol):
    """
    Cache selection decisions. Optional optimization.

    Agent can implement to avoid redundant LLM calls for similar queries.
    """

    async def get(self, query_hash: str) -> SelectionResult | None:
        """Get cached selection for query."""
        ...

    async def put(self, query_hash: str, result: SelectionResult, ttl: int = 3600) -> None:
        """Cache selection result."""
        ...

    async def invalidate(self, fact_ids: list[int]) -> None:
        """Invalidate cache entries that used these facts."""
        ...
```

---

## Agent Implements: Selection Logic

Agent provides concrete implementations of the protocols.

### Example: Rule-Based Selector

```python
# agent/adaptation/selectors.py

class RuleBasedSelector(ContextSelector):
    """Fast rule-based selection for simple cases."""

    def __init__(self, rules: list[SelectionRule]):
        self.rules = rules

    async def select(
        self,
        query: str,
        candidates: list[ScoredFact],
        max_facts: int = 10,
    ) -> SelectionResult:
        selected = []

        for candidate in candidates:
            # Always include high-similarity facts
            if candidate.similarity > 0.85:
                selected.append(candidate.fact)
                continue

            # Check rule matches
            for rule in self.rules:
                if rule.matches(query, candidate.fact):
                    selected.append(candidate.fact)
                    break

            if len(selected) >= max_facts:
                break

        return SelectionResult(
            facts=selected,
            confidence=0.9 if selected else 0.3,
            tier="rules",
            candidates_considered=len(candidates),
        )
```

### Example: LLM-Based Selector

```python
class LLMSelector(ContextSelector):
    """Use small LLM to score and select facts."""

    SYSTEM_PROMPT = """
    You are a context selection assistant. Given a user query and candidate facts,
    select the facts most relevant to answering the query.

    Output JSON: {"selected_ids": [1, 3, 7], "reasoning": "..."}

    Selection criteria:
    - Directly relevant to the query topic
    - Provides necessary background for accurate response
    - Respects user preferences that affect response style
    """

    def __init__(self, model: LLMBackend):
        self.model = model

    async def select(
        self,
        query: str,
        candidates: list[ScoredFact],
        max_facts: int = 10,
    ) -> SelectionResult:
        # Score each candidate with LLM
        scores = await self._batch_score(query, candidates)

        # Select top scoring
        sorted_candidates = sorted(
            zip(candidates, scores),
            key=lambda x: x[1],
            reverse=True,
        )
        selected = [c.fact for c, s in sorted_candidates[:max_facts]]

        avg_score = sum(scores[:max_facts]) / max_facts if scores else 0

        return SelectionResult(
            facts=selected,
            confidence=avg_score,
            tier="llm",
            reasoning=f"Selected {len(selected)} facts with avg score {avg_score:.2f}",
            candidates_considered=len(candidates),
        )

    async def _batch_score(
        self,
        query: str,
        candidates: list[ScoredFact],
    ) -> list[float]:
        """Score all candidates for relevance to query."""
        prompt = self._build_scoring_prompt(query, candidates)
        response = await self.model.complete(self.SYSTEM_PROMPT, prompt)
        return self._parse_scores(response)
```

### Example: Hybrid Selector (Three-Tier)

The recommended approach: escalate through tiers based on confidence.

```python
class HybridSelector(ContextSelector):
    """
    Three-tier selection: rules → embeddings → LLM.

    Escalates through tiers based on confidence.
    """

    def __init__(
        self,
        rule_selector: RuleBasedSelector,
        llm_selector: LLMSelector,
        rule_confidence_threshold: float = 0.8,
        embedding_confidence_threshold: float = 0.75,
    ):
        self.rules = rule_selector
        self.llm = llm_selector
        self.rule_threshold = rule_confidence_threshold
        self.embedding_threshold = embedding_confidence_threshold

    async def select(
        self,
        query: str,
        candidates: list[ScoredFact],
        max_facts: int = 10,
    ) -> SelectionResult:
        # Tier 1: Try rules first (fast, free)
        result = await self.rules.select(query, candidates, max_facts)
        if result.confidence >= self.rule_threshold:
            return result

        # Tier 2: Check embedding scores (fast, free)
        high_similarity = [c for c in candidates if c.similarity > 0.8]
        if len(high_similarity) >= max_facts * 0.7:
            return SelectionResult(
                facts=[c.fact for c in high_similarity[:max_facts]],
                confidence=self.embedding_threshold,
                tier="embedding",
                candidates_considered=len(candidates),
            )

        # Tier 3: Use LLM (slower, costs tokens)
        return await self.llm.select(query, candidates, max_facts)
```

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
│  │ • Check similarity scores from kelt's vector search      │  │
│  │ • High confidence if top results clearly separate         │  │
│  │ • Return top-k if confident                               │  │
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
```

---

## Pipeline Integration

Kelt's pipeline uses the agent-provided selector:

```python
# kelt/adaptation/pipeline.py

class AdaptationPipeline:
    def __init__(
        self,
        kelt_client: Client,
        context_selector: ContextSelector,  # Agent provides this
        ...
    ):
        self.kelt = kelt_client
        self.selector = context_selector

    async def process(self, query: str, messages: list, system: str | None) -> PipelineResult:
        # Step 1: Kelt retrieves candidates
        candidates = await self.kelt.content.search_similar(
            query,
            top_k=self.config.candidate_limit,
        )

        # Step 2: Agent selects from candidates
        selection = await self.selector.select(query, candidates)

        # Step 3: Build context string (kelt)
        context_str = self.kelt.build_context([f.id for f in selection.facts])

        # ... continue pipeline
```

---

## Selection Rules (Agent-Defined)

Agent can define rules for fast context routing:

```python
# agent/adaptation/rules.py

@dataclass
class SelectionRule:
    """Rule for context selection."""
    name: str
    condition: Callable[[str, Fact], bool]
    priority: int = 0
    always_include: bool = False

# Example rules
RULES = [
    SelectionRule(
        name="always_include_identity",
        condition=lambda q, f: f.category == "identity",
        always_include=True,
    ),
    SelectionRule(
        name="preferences_for_how_questions",
        condition=lambda q, f: "how" in q.lower() and f.category == "preferences",
        priority=10,
    ),
    SelectionRule(
        name="work_context_for_project_queries",
        condition=lambda q, f: "project" in q.lower() and f.category == "work",
        priority=5,
    ),
]
```

Or configuration-driven rules:

```yaml
# agent.yaml
context_selection:
  rules:
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

---

## Feedback Tracking (Optional)

Kelt can optionally store routing decisions for analysis:

```sql
-- Optional: Track routing decisions for learning
CREATE TABLE routing_decisions (
    id BIGSERIAL PRIMARY KEY,
    context_key VARCHAR(100),
    query_hash VARCHAR(64) NOT NULL,
    tier_used VARCHAR(20) NOT NULL,  -- 'rules', 'embedding', 'llm'
    facts_selected BIGINT[] NOT NULL,
    candidates_count INT,
    confidence REAL NOT NULL,
    reasoning TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE routing_feedback (
    id BIGSERIAL PRIMARY KEY,
    decision_id BIGINT NOT NULL REFERENCES routing_decisions(id),
    signal VARCHAR(20) NOT NULL,  -- 'positive', 'negative', 'missing_context'
    comment TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

Agent can use this data to improve selection over time.

---

## Configuration

```yaml
# llm-kelt.yaml - kelt's config
adaptation:
  candidate_limit: 30           # Max candidates from vector search
  min_similarity: 0.5           # Minimum embedding similarity

# agent.yaml - agent's config
context_selection:
  rule_confidence_threshold: 0.8
  embedding_confidence_threshold: 0.75
  llm_selector:
    model: "small"
    batch_size: 10
  cache:
    enabled: true
    ttl: 3600
```

---

## Summary

| Component | Owner | Responsibility |
|-----------|-------|---------------|
| Vector search | kelt | Retrieve candidates by embedding similarity |
| Fact retrieval | kelt | List facts by category/confidence |
| `ContextSelector` protocol | kelt | Define what agent must implement |
| `SelectionResult` type | kelt | Standardized result format |
| Rule-based selection | agent | Fast selection via rules |
| LLM-based selection | agent | Accurate selection via model |
| Hybrid selection | agent | Tiered approach (rules → embeddings → LLM) |
| Routing cache | agent | Optional caching of decisions |

Kelt provides the primitives and protocols. Agent provides the intelligence.
