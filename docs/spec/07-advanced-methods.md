# Advanced Adaptation Methods

## Problem

Simple strategy composition (Memory + RAG + LoRA) is insufficient for intelligent adaptation:

```python
# Current: Linear composition
class CompositeStrategy:
    async def adapt_request(self, messages, system):
        for strategy in self.strategies:
            request = await strategy.adapt_request(messages, system)
        return request  # No feedback, no validation, no branching
```

Real-world needs:
- **Validate before returning** - Check response quality, retry if poor
- **Conditional paths** - Different strategies based on query type or confidence
- **Iterative refinement** - Multiple passes until quality threshold
- **Dynamic routing** - LLM decides what adaptation to apply

## Solution: Adaptation Pipelines

Pipelines are directed graphs of steps with branching, loops, and quality gates.

### Core Concepts

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Pipeline Execution                             │
│                                                                          │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐         │
│   │  Step 1  │───▶│  Step 2  │───▶│  Gate    │───▶│  Step 3  │         │
│   │ (Retrieve)│    │(Generate)│    │ (Score)  │    │ (Return) │         │
│   └──────────┘    └──────────┘    └────┬─────┘    └──────────┘         │
│                                        │                                 │
│                                   score < 0.7                            │
│                                        │                                 │
│                                        ▼                                 │
│                                  ┌──────────┐                           │
│                                  │  Step 2b │                           │
│                                  │ (Refine) │──────────────┐            │
│                                  └──────────┘              │            │
│                                        │                   │            │
│                                        └───────────────────┘            │
│                                          (retry up to N)                │
└─────────────────────────────────────────────────────────────────────────┘
```

### Step Types

| Type | Purpose | Example |
|------|---------|---------|
| **Retrieval** | Fetch context | RAG search, memory lookup, fact retrieval |
| **Generation** | LLM inference | Main response, summary, critique |
| **Evaluation** | Score/validate | Quality check, relevance scoring, safety filter |
| **Transform** | Modify data | Merge contexts, reformat, filter |
| **Route** | Conditional branch | Query classification, confidence threshold |
| **Loop** | Iterate until condition | Retry with backoff, refinement cycles |

## Data Model

### Pipeline Definition

```python
@dataclass
class PipelineStep:
    id: str
    type: StepType  # retrieval, generation, evaluation, transform, route, loop
    config: dict    # Step-specific configuration
    next: str | dict[str, str] | None  # Next step or conditional mapping


@dataclass
class Pipeline:
    id: str
    name: str
    steps: dict[str, PipelineStep]  # step_id -> step
    entry: str                       # Entry step id

    def validate(self) -> list[str]:
        """Validate pipeline structure (no orphans, valid references)."""
        ...
```

### Execution State

```python
@dataclass
class PipelineState:
    """Carries context through pipeline execution."""

    # Input
    query: str
    messages: list[Message]
    system: str | None

    # Accumulated context
    retrieved_facts: list[Fact]
    retrieved_documents: list[Document]

    # Generation artifacts
    response: str | None
    intermediate_responses: list[str]

    # Evaluation results
    scores: dict[str, float]  # evaluator_id -> score

    # Execution metadata
    steps_executed: list[str]
    retry_count: int

    def clone(self) -> "PipelineState":
        """Create copy for branching."""
        ...
```

## Step Implementations

### Retrieval Steps

```python
class FactRetrievalStep:
    """Retrieve facts from memory store."""

    async def execute(self, state: PipelineState, ctx: PipelineContext) -> PipelineState:
        facts = await ctx.fact_store.retrieve(
            query=state.query,
            profile_id=ctx.profile_id,
            top_k=self.config.get("top_k", 10),
            min_similarity=self.config.get("min_similarity", 0.7),
        )
        state.retrieved_facts.extend(facts)
        return state


class RAGRetrievalStep:
    """Retrieve documents from vector store."""

    async def execute(self, state: PipelineState, ctx: PipelineContext) -> PipelineState:
        docs = await ctx.vector_store.search(
            query=state.query,
            collection=self.config["collection"],
            top_k=self.config.get("top_k", 5),
        )
        state.retrieved_documents.extend(docs)
        return state
```

### Generation Steps

```python
class GenerationStep:
    """Generate response with accumulated context."""

    async def execute(self, state: PipelineState, ctx: PipelineContext) -> PipelineState:
        # Build context from retrieved artifacts
        context = self._build_context(state)

        # Generate
        response = await ctx.llm.generate(
            messages=state.messages,
            system=self._inject_context(state.system, context),
            temperature=self.config.get("temperature", 0.7),
            max_tokens=self.config.get("max_tokens"),
        )

        state.response = response.content
        state.intermediate_responses.append(response.content)
        return state
```

### Evaluation Steps

```python
class QualityEvaluator:
    """Score response quality."""

    async def execute(self, state: PipelineState, ctx: PipelineContext) -> PipelineState:
        score = await self._evaluate(state, ctx)
        state.scores[self.id] = score
        return state

    async def _evaluate(self, state: PipelineState, ctx: PipelineContext) -> float:
        """Override in subclasses."""
        raise NotImplementedError


class LLMJudgeEvaluator(QualityEvaluator):
    """Use LLM to judge response quality."""

    JUDGE_PROMPT = """
    Rate the response quality from 0.0 to 1.0.

    Query: {query}
    Context provided: {context}
    Response: {response}

    Criteria:
    - Relevance to query
    - Use of provided context
    - Factual consistency
    - Completeness

    Output only a number between 0.0 and 1.0.
    """

    async def _evaluate(self, state: PipelineState, ctx: PipelineContext) -> float:
        prompt = self.JUDGE_PROMPT.format(
            query=state.query,
            context=self._format_context(state),
            response=state.response,
        )
        result = await ctx.judge_llm.generate(prompt)
        return float(result.content.strip())


class EmbeddingSimilarityEvaluator(QualityEvaluator):
    """Score based on embedding similarity between query and response."""

    async def _evaluate(self, state: PipelineState, ctx: PipelineContext) -> float:
        query_emb = await ctx.embedder.embed(state.query)
        response_emb = await ctx.embedder.embed(state.response)
        return cosine_similarity(query_emb, response_emb)


class RuleBasedEvaluator(QualityEvaluator):
    """Score based on deterministic rules."""

    async def _evaluate(self, state: PipelineState, ctx: PipelineContext) -> float:
        score = 1.0

        # Check length
        if len(state.response) < self.config.get("min_length", 10):
            score -= 0.3

        # Check context usage
        if state.retrieved_facts and not self._uses_facts(state):
            score -= 0.2

        # Check for refusals
        if self._is_refusal(state.response):
            score -= 0.5

        return max(0.0, score)
```

### Routing Steps

```python
class ThresholdRouter:
    """Route based on score threshold."""

    def __init__(self, config: dict):
        self.score_key = config["score_key"]
        self.threshold = config["threshold"]
        self.above = config["above"]  # Next step if score >= threshold
        self.below = config["below"]  # Next step if score < threshold

    def route(self, state: PipelineState) -> str:
        score = state.scores.get(self.score_key, 0.0)
        return self.above if score >= self.threshold else self.below


class QueryClassifierRouter:
    """Route based on query classification."""

    CATEGORIES = ["factual", "creative", "coding", "conversation"]

    async def route(self, state: PipelineState, ctx: PipelineContext) -> str:
        category = await self._classify(state.query, ctx)
        return self.config["routes"].get(category, self.config["default"])

    async def _classify(self, query: str, ctx: PipelineContext) -> str:
        # Could be rule-based, embedding-based, or LLM-based
        ...
```

### Loop Steps

```python
class RetryLoop:
    """Retry a subpipeline until condition met."""

    def __init__(self, config: dict):
        self.max_retries = config.get("max_retries", 3)
        self.score_key = config["score_key"]
        self.threshold = config["threshold"]
        self.retry_step = config["retry_step"]  # Step to jump to on retry
        self.success_step = config["success_step"]
        self.failure_step = config["failure_step"]

    def evaluate(self, state: PipelineState) -> str:
        score = state.scores.get(self.score_key, 0.0)

        if score >= self.threshold:
            return self.success_step

        if state.retry_count >= self.max_retries:
            return self.failure_step

        state.retry_count += 1
        return self.retry_step
```

## Pipeline Executor

```python
class PipelineExecutor:
    """Execute adaptation pipelines."""

    def __init__(
        self,
        llm: LLMClient,
        judge_llm: LLMClient | None,
        fact_store: FactStore,
        vector_store: VectorStore | None,
        embedder: Embedder,
    ):
        self.context = PipelineContext(
            llm=llm,
            judge_llm=judge_llm or llm,
            fact_store=fact_store,
            vector_store=vector_store,
            embedder=embedder,
        )
        self.step_registry: dict[str, type] = {}

    async def execute(
        self,
        pipeline: Pipeline,
        query: str,
        messages: list[Message],
        system: str | None,
        profile_id: int,
        timeout: float = 30.0,
    ) -> PipelineResult:
        """Execute pipeline and return result."""

        state = PipelineState(
            query=query,
            messages=messages,
            system=system,
            retrieved_facts=[],
            retrieved_documents=[],
            response=None,
            intermediate_responses=[],
            scores={},
            steps_executed=[],
            retry_count=0,
        )

        self.context.profile_id = profile_id
        current_step = pipeline.entry

        async with asyncio.timeout(timeout):
            while current_step:
                step_def = pipeline.steps[current_step]
                step = self._instantiate_step(step_def)

                state.steps_executed.append(current_step)
                state = await step.execute(state, self.context)

                # Determine next step
                if isinstance(step_def.next, str):
                    current_step = step_def.next
                elif isinstance(step_def.next, dict):
                    # Conditional routing
                    current_step = step.route(state)
                else:
                    current_step = None  # Terminal step

        return PipelineResult(
            response=state.response,
            facts_used=state.retrieved_facts,
            documents_used=state.retrieved_documents,
            scores=state.scores,
            steps_executed=state.steps_executed,
            retries=state.retry_count,
        )
```

## Example Pipelines

### Basic: RAG + Memory with Validation

```yaml
pipeline:
  name: "rag_memory_validated"
  entry: "retrieve_facts"

  steps:
    retrieve_facts:
      type: fact_retrieval
      config:
        top_k: 10
        min_similarity: 0.7
      next: "retrieve_docs"

    retrieve_docs:
      type: rag_retrieval
      config:
        collection: "knowledge_base"
        top_k: 5
      next: "generate"

    generate:
      type: generation
      config:
        temperature: 0.7
      next: "evaluate"

    evaluate:
      type: llm_judge
      config:
        criteria: ["relevance", "factuality", "completeness"]
      next: "quality_gate"

    quality_gate:
      type: threshold_router
      config:
        score_key: "evaluate"
        threshold: 0.7
        above: "return"
        below: "refine"

    refine:
      type: retry_loop
      config:
        max_retries: 2
        score_key: "evaluate"
        threshold: 0.7
        retry_step: "generate"  # Re-generate with same context
        success_step: "return"
        failure_step: "return_with_warning"

    return:
      type: terminal
      config:
        output: "response"

    return_with_warning:
      type: terminal
      config:
        output: "response"
        metadata:
          low_confidence: true
```

### Advanced: Query-Adaptive Pipeline

```yaml
pipeline:
  name: "query_adaptive"
  entry: "classify"

  steps:
    classify:
      type: query_classifier
      config:
        routes:
          factual: "heavy_retrieval"
          creative: "light_context"
          coding: "code_retrieval"
          conversation: "memory_only"
        default: "balanced"

    # Factual queries: heavy retrieval, strict validation
    heavy_retrieval:
      type: composite
      steps:
        - type: fact_retrieval
          config: { top_k: 20, min_similarity: 0.8 }
        - type: rag_retrieval
          config: { top_k: 10 }
      next: "strict_generate"

    strict_generate:
      type: generation
      config:
        temperature: 0.3
        system_suffix: "Cite sources. If unsure, say so."
      next: "strict_evaluate"

    strict_evaluate:
      type: llm_judge
      config:
        criteria: ["factuality", "citations"]
        threshold: 0.8
      next: "strict_gate"

    strict_gate:
      type: threshold_router
      config:
        score_key: "strict_evaluate"
        threshold: 0.8
        above: "return"
        below: "heavy_retrieval"  # Retry with more context

    # Creative queries: light context, flexible validation
    light_context:
      type: fact_retrieval
      config:
        top_k: 5
        categories: ["preferences", "style"]
      next: "creative_generate"

    creative_generate:
      type: generation
      config:
        temperature: 0.9
      next: "return"  # No validation for creative

    # ... other branches
```

### Iterative Refinement

```yaml
pipeline:
  name: "iterative_refinement"
  entry: "initial_generate"

  steps:
    initial_generate:
      type: generation
      config:
        temperature: 0.7
      next: "critique"

    critique:
      type: generation
      config:
        role: "critic"
        system: |
          Review this response and identify weaknesses:
          - Missing information
          - Inaccuracies
          - Unclear explanations
          Output a critique, or "APPROVED" if satisfactory.
        input_from: "response"
      next: "check_critique"

    check_critique:
      type: rule_router
      config:
        condition: "critique_contains_approved"
        true: "return"
        false: "refine"

    refine:
      type: generation
      config:
        system: |
          Improve your response based on this critique:
          {critique}
        temperature: 0.5
      next: "refinement_loop"

    refinement_loop:
      type: retry_loop
      config:
        max_retries: 2
        retry_step: "critique"
        success_step: "return"
        failure_step: "return"
```

## Integration with Serving

```python
class PipelineOrchestrator:
    """Orchestrate pipeline execution for serving."""

    def __init__(
        self,
        executor: PipelineExecutor,
        pipeline_store: PipelineStore,
        default_pipeline: str = "basic",
    ):
        self.executor = executor
        self.pipeline_store = pipeline_store
        self.default_pipeline = default_pipeline

    async def adapt(
        self,
        profile_id: int,
        query: str,
        messages: list[Message],
        system: str | None,
        pipeline_id: str | None = None,
    ) -> AdaptationResult:
        """Run adaptation pipeline for request."""

        # Get pipeline (profile-specific or default)
        pipeline = await self._get_pipeline(profile_id, pipeline_id)

        # Execute
        result = await self.executor.execute(
            pipeline=pipeline,
            query=query,
            messages=messages,
            system=system,
            profile_id=profile_id,
        )

        return AdaptationResult(
            messages=messages,
            system=self._build_system(system, result),
            metadata={
                "pipeline": pipeline.name,
                "steps": result.steps_executed,
                "scores": result.scores,
                "retries": result.retries,
            },
        )
```

## Pipeline Storage

```sql
CREATE TABLE pipelines (
    id BIGSERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL UNIQUE,
    description TEXT,
    definition JSONB NOT NULL,  -- Full pipeline YAML/JSON
    is_default BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE profile_pipelines (
    id BIGSERIAL PRIMARY KEY,
    profile_id BIGINT NOT NULL REFERENCES profiles(id),
    pipeline_id BIGINT NOT NULL REFERENCES pipelines(id),
    query_pattern VARCHAR(255),  -- Optional: use this pipeline for matching queries
    priority INT DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(profile_id, pipeline_id, query_pattern)
);

CREATE TABLE pipeline_executions (
    id BIGSERIAL PRIMARY KEY,
    profile_id BIGINT NOT NULL REFERENCES profiles(id),
    pipeline_id BIGINT NOT NULL REFERENCES pipelines(id),
    query_hash VARCHAR(64) NOT NULL,
    steps_executed TEXT[] NOT NULL,
    scores JSONB,
    retries INT DEFAULT 0,
    duration_ms INT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

## Open Questions

1. **Pipeline authoring**: How do users create/modify pipelines? YAML? UI? Code?
2. **Debugging**: How to debug pipeline execution? Logging? Visualization?
3. **Cost tracking**: How to track/limit LLM calls in complex pipelines?
4. **Caching**: Cache intermediate results? At which granularity?
5. **Parallelism**: Support parallel step execution (e.g., multiple retrievers)?
6. **Versioning**: How to version pipelines and handle migrations?
