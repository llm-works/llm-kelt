# Advanced Adaptation Methods

Learn's pipeline framework with hooks for agent-provided intelligence.

## Problem

Simple strategy composition (Memory + RAG + LoRA) is insufficient for intelligent adaptation:

```python
# Linear composition - no validation, no branching, no retry
class CompositeStrategy:
    async def adapt_request(self, messages, system):
        for strategy in self.strategies:
            request = await strategy.adapt_request(messages, system)
        return request  # What if response is bad? No feedback loop.
```

Real-world needs:
- **Validate before returning** - Check response quality, retry if poor
- **Conditional paths** - Different strategies based on query type or confidence
- **Iterative refinement** - Multiple passes until quality threshold
- **Dynamic routing** - Smart decisions about context and models

## Solution: Pipeline Framework

Learn provides the **pipeline executor** (structure, flow, data types).
Agent provides the **intelligence** (selectors, evaluators, routers, retry strategies).

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Pipeline Execution                             │
│                                                                          │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐         │
│   │ Retrieve │───▶│ Generate │───▶│ Evaluate │───▶│  Return  │         │
│   │ (learn)  │    │ (agent)  │    │ (agent)  │    │          │         │
│   └──────────┘    └──────────┘    └────┬─────┘    └──────────┘         │
│                                        │                                 │
│                                   score < 0.7                            │
│                                        │                                 │
│                                        ▼                                 │
│                                  ┌──────────┐                           │
│                                  │  Retry   │                           │
│                                  │ (agent)  │──────────────┐            │
│                                  └──────────┘              │            │
│                                        │                   │            │
│                                        └───────────────────┘            │
│                                          (retry up to N)                │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Learn Provides: Pipeline Framework

### Data Types

```python
# learn/adaptation/types.py

@dataclass
class PipelineState:
    """Carries context through pipeline execution."""

    # Input
    query: str
    messages: list[Message]
    system: str | None

    # Accumulated context (learn populates)
    retrieved_facts: list[Fact]
    candidates: list[ScoredFact]

    # Selection result (agent populates via selector)
    selected_facts: list[Fact]
    selection_confidence: float

    # Generation artifacts
    response: str | None
    intermediate_responses: list[str]

    # Evaluation results (agent populates via evaluator)
    scores: dict[str, float]

    # Execution metadata
    model_used: str | None
    attempts: int
    total_tokens: int


@dataclass
class PipelineResult:
    """Final result from pipeline execution."""
    response: str
    facts_used: list[Fact]
    model_used: str
    quality_score: float
    attempts: int
    total_tokens: int


@dataclass
class PipelineConfig:
    """Pipeline configuration."""
    max_attempts: int = 3
    candidate_limit: int = 30
    quality_threshold: float = 0.7
    timeout: float = 30.0
```

### Protocol Definitions

```python
# learn/adaptation/protocols.py

from typing import Protocol, runtime_checkable

@runtime_checkable
class ContextSelector(Protocol):
    """Select relevant context. See 06-context-routing.md."""
    async def select(self, query: str, candidates: list[ScoredFact], max_facts: int) -> SelectionResult:
        ...


@runtime_checkable
class QualityEvaluator(Protocol):
    """Evaluate response quality. Agent implements this."""

    async def evaluate(
        self,
        query: str,
        response: str,
        context: list[Fact],
    ) -> EvaluationResult:
        """
        Score response quality.

        Args:
            query: Original user query
            response: Generated response
            context: Facts used in generation

        Returns:
            EvaluationResult with score (0-1), passed flag, reasoning
        """
        ...


@runtime_checkable
class ModelRouter(Protocol):
    """Decide which model to use. Agent implements this."""

    async def route(
        self,
        query: str,
        context_confidence: float,
    ) -> RoutingDecision:
        """
        Decide: use small model (fast/cheap) or large model (powerful)?

        Args:
            query: User query
            context_confidence: Confidence from context selection

        Returns:
            RoutingDecision with model identifier and reasoning
        """
        ...


@runtime_checkable
class RetryStrategy(Protocol):
    """Decide how to handle failures. Agent implements this."""

    async def should_retry(
        self,
        attempt: int,
        quality_score: float,
        error: Exception | None,
    ) -> RetryDecision:
        """
        Decide whether to retry, escalate, or give up.

        Args:
            attempt: Current attempt number (1-indexed)
            quality_score: Score from quality evaluator
            error: Exception if generation failed

        Returns:
            RetryDecision with retry/escalate/give_up flags
        """
        ...
```

### Result Types

```python
# learn/adaptation/types.py

@dataclass
class EvaluationResult:
    """Result from quality evaluation."""
    score: float  # 0-1
    passed: bool  # score >= threshold
    reasoning: str | None = None
    criteria: dict[str, float] | None = None  # Per-criterion scores


@dataclass
class RoutingDecision:
    """Result from model routing."""
    model: str  # Model identifier
    reason: str  # Why this model


@dataclass
class RetryDecision:
    """Result from retry strategy."""
    should_retry: bool
    escalate_model: bool  # Try larger model?
    adjust_context: bool  # Retrieve more context?
    give_up: bool
```

### Pipeline Executor

```python
# learn/adaptation/pipeline.py

class AdaptationPipeline:
    """
    Pipeline executor. Learn controls flow, agent provides intelligence.
    """

    def __init__(
        self,
        learn_client: LearnClient,
        # Agent-provided implementations
        context_selector: ContextSelector,
        quality_evaluator: QualityEvaluator,
        model_router: ModelRouter,
        retry_strategy: RetryStrategy,
        # Model backends
        models: dict[str, LLMBackend],
        config: PipelineConfig,
    ):
        self.learn = learn_client
        self.selector = context_selector
        self.evaluator = quality_evaluator
        self.router = model_router
        self.retry = retry_strategy
        self.models = models
        self.config = config

    async def process(
        self,
        query: str,
        messages: list[Message],
        system: str | None = None,
    ) -> PipelineResult:
        """Execute the adaptation pipeline."""

        state = PipelineState(
            query=query,
            messages=messages,
            system=system,
            retrieved_facts=[],
            candidates=[],
            selected_facts=[],
            selection_confidence=0.0,
            response=None,
            intermediate_responses=[],
            scores={},
            model_used=None,
            attempts=0,
            total_tokens=0,
        )

        async with asyncio.timeout(self.config.timeout):
            while state.attempts < self.config.max_attempts:
                state.attempts += 1

                # Step 1: Retrieve candidates (learn primitive)
                state.candidates = await self.learn.content.search_similar(
                    query,
                    top_k=self.config.candidate_limit,
                )

                # Step 2: Select context (AGENT HOOK)
                selection = await self.selector.select(
                    query, state.candidates, max_facts=10
                )
                state.selected_facts = selection.facts
                state.selection_confidence = selection.confidence

                # Step 3: Route to model (AGENT HOOK)
                routing = await self.router.route(query, selection.confidence)
                model = self.models[routing.model]
                state.model_used = routing.model

                # Step 4: Build context and generate (learn primitive)
                context_str = self.learn.build_context(
                    [f.id for f in selection.facts]
                )
                full_system = f"{system or ''}\n\n{context_str}".strip()

                result = await model.complete(full_system, messages)
                state.response = result.content
                state.intermediate_responses.append(result.content)
                state.total_tokens += result.tokens

                # Step 5: Evaluate quality (AGENT HOOK)
                evaluation = await self.evaluator.evaluate(
                    query, result.content, selection.facts
                )
                state.scores["quality"] = evaluation.score

                # Step 6: Check if done or retry (AGENT HOOK)
                if evaluation.passed:
                    await self._record_success(state, evaluation)
                    return self._build_result(state, evaluation)

                # Failed - ask retry strategy
                retry_decision = await self.retry.should_retry(
                    state.attempts, evaluation.score, error=None
                )

                if retry_decision.give_up:
                    break

                if retry_decision.escalate_model:
                    # Force larger model on next attempt
                    self.router = _ForcedRouter(self.config.large_model)

        # Max attempts reached or gave up
        await self._record_failure(state)
        return self._build_result(state, evaluation)

    def _build_result(self, state: PipelineState, evaluation: EvaluationResult) -> PipelineResult:
        return PipelineResult(
            response=state.response or "",
            facts_used=state.selected_facts,
            model_used=state.model_used or "unknown",
            quality_score=evaluation.score,
            attempts=state.attempts,
            total_tokens=state.total_tokens,
        )

    async def _record_success(self, state: PipelineState, evaluation: EvaluationResult):
        """Record successful interaction to learn."""
        await self.learn.feedback.record(
            signal="positive",
            content_text=state.response,
            context={
                "query": state.query,
                "facts_used": [f.id for f in state.selected_facts],
                "quality_score": evaluation.score,
                "model_used": state.model_used,
            },
        )

    async def _record_failure(self, state: PipelineState):
        """Record failed interaction."""
        await self.learn.feedback.record(
            signal="negative",
            content_text=state.query,
            context={
                "attempts": state.attempts,
                "reason": "max_attempts_exceeded",
            },
        )
```

---

## Agent Implements: Intelligence

Agent provides concrete implementations of learn's protocols.

### Example: Quality Evaluators

```python
# agent/adaptation/evaluators.py

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

    def __init__(self, judge_model: LLMBackend, threshold: float = 0.7):
        self.model = judge_model
        self.threshold = threshold

    async def evaluate(
        self,
        query: str,
        response: str,
        context: list[Fact],
    ) -> EvaluationResult:
        prompt = self.JUDGE_PROMPT.format(
            query=query,
            context="\n".join(f.content for f in context),
            response=response,
        )
        result = await self.model.complete("You are a quality judge.", prompt)
        score = float(result.content.strip())

        return EvaluationResult(
            score=score,
            passed=score >= self.threshold,
            reasoning=f"LLM judge score: {score:.2f}",
        )


class RuleBasedEvaluator(QualityEvaluator):
    """Fast rule-based quality checks."""

    def __init__(self, threshold: float = 0.7, min_length: int = 20):
        self.threshold = threshold
        self.min_length = min_length

    async def evaluate(
        self,
        query: str,
        response: str,
        context: list[Fact],
    ) -> EvaluationResult:
        score = 1.0
        criteria = {}

        # Check length
        if len(response) < self.min_length:
            score -= 0.3
            criteria["length"] = 0.0
        else:
            criteria["length"] = 1.0

        # Check for refusals
        refusal_phrases = ["I don't know", "I cannot", "I'm not able"]
        if any(phrase in response for phrase in refusal_phrases):
            score -= 0.4
            criteria["refusal"] = 0.0
        else:
            criteria["refusal"] = 1.0

        # Check context usage
        if context and not self._uses_context(response, context):
            score -= 0.2
            criteria["context_usage"] = 0.0
        else:
            criteria["context_usage"] = 1.0

        final_score = max(0.0, score)

        return EvaluationResult(
            score=final_score,
            passed=final_score >= self.threshold,
            criteria=criteria,
        )

    def _uses_context(self, response: str, context: list[Fact]) -> bool:
        """Check if response uses any context."""
        response_lower = response.lower()
        for fact in context:
            # Simple keyword check
            words = fact.content.lower().split()
            if any(word in response_lower for word in words if len(word) > 4):
                return True
        return False
```

### Example: Model Routers

```python
# agent/adaptation/routers.py

class ConfidenceBasedRouter(ModelRouter):
    """Route based on context selection confidence."""

    def __init__(
        self,
        small_model: str,
        large_model: str,
        threshold: float = 0.7,
    ):
        self.small = small_model
        self.large = large_model
        self.threshold = threshold

    async def route(
        self,
        query: str,
        context_confidence: float,
    ) -> RoutingDecision:
        if context_confidence >= self.threshold:
            return RoutingDecision(
                model=self.small,
                reason=f"High confidence ({context_confidence:.2f}) - using small model",
            )
        else:
            return RoutingDecision(
                model=self.large,
                reason=f"Low confidence ({context_confidence:.2f}) - escalating to large model",
            )


class QueryComplexityRouter(ModelRouter):
    """Route based on query complexity analysis."""

    def __init__(self, small_model: str, large_model: str, analyzer: LLMBackend):
        self.small = small_model
        self.large = large_model
        self.analyzer = analyzer

    async def route(
        self,
        query: str,
        context_confidence: float,
    ) -> RoutingDecision:
        complexity = await self._analyze_complexity(query)

        if complexity == "simple":
            return RoutingDecision(model=self.small, reason="Simple query")
        elif complexity == "complex":
            return RoutingDecision(model=self.large, reason="Complex reasoning required")
        else:
            # Medium - use context confidence as tiebreaker
            if context_confidence >= 0.7:
                return RoutingDecision(model=self.small, reason="Medium query, good context")
            else:
                return RoutingDecision(model=self.large, reason="Medium query, weak context")
```

### Example: Retry Strategies

```python
# agent/adaptation/retry.py

class ExponentialBackoffRetry(RetryStrategy):
    """Retry with escalation on repeated failures."""

    def __init__(
        self,
        max_attempts: int = 3,
        quality_threshold: float = 0.7,
        escalate_after: int = 2,
    ):
        self.max_attempts = max_attempts
        self.threshold = quality_threshold
        self.escalate_after = escalate_after

    async def should_retry(
        self,
        attempt: int,
        quality_score: float,
        error: Exception | None,
    ) -> RetryDecision:
        # Give up if max attempts reached
        if attempt >= self.max_attempts:
            return RetryDecision(
                should_retry=False,
                escalate_model=False,
                adjust_context=False,
                give_up=True,
            )

        # Give up if score is very low (hopeless)
        if quality_score < 0.3:
            return RetryDecision(
                should_retry=False,
                escalate_model=False,
                adjust_context=False,
                give_up=True,
            )

        # Escalate model after threshold attempts
        escalate = attempt >= self.escalate_after

        return RetryDecision(
            should_retry=True,
            escalate_model=escalate,
            adjust_context=True,  # Try more context
            give_up=False,
        )


class ConservativeRetry(RetryStrategy):
    """Conservative: only retry on errors, not low quality."""

    async def should_retry(
        self,
        attempt: int,
        quality_score: float,
        error: Exception | None,
    ) -> RetryDecision:
        # Only retry on actual errors
        if error is not None and attempt < 3:
            return RetryDecision(
                should_retry=True,
                escalate_model=False,
                adjust_context=False,
                give_up=False,
            )

        # Accept whatever quality we got
        return RetryDecision(
            should_retry=False,
            escalate_model=False,
            adjust_context=False,
            give_up=True,
        )
```

---

## Agent: Wiring It Together

```python
# agent/adaptation/orchestrator.py

from llm_kelt import LearnClient
from llm_kelt.adaptation import AdaptationPipeline, PipelineConfig

from agent.adaptation.selectors import HybridSelector
from agent.adaptation.evaluators import LLMJudgeEvaluator
from agent.adaptation.routers import ConfidenceBasedRouter
from agent.adaptation.retry import ExponentialBackoffRetry

class OrchestrationAgent(WorkerAgent):
    """Agent that uses learn's pipeline with custom intelligence."""

    def setup(self):
        super().setup()

        # Learn client for data primitives
        self.learn = LearnClient(context_key=self.config.context_key)

        # Models
        self.small_model = LocalBackend(self.config.small_model_url)
        self.large_model = LocalBackend(self.config.large_model_url)

        # Agent's implementations of learn's protocols
        selector = HybridSelector(
            rule_selector=RuleBasedSelector(SELECTION_RULES),
            llm_selector=LLMSelector(self.small_model),
        )
        evaluator = LLMJudgeEvaluator(self.small_model, threshold=0.7)
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
            config=PipelineConfig(max_attempts=3, quality_threshold=0.7),
        )

    async def process_query(self, query: str, messages: list[Message]) -> str:
        result = await self.pipeline.process(query, messages)
        return result.response
```

---

## Pipeline Storage (Optional)

Learn can optionally store pipeline execution history:

```sql
CREATE TABLE pipeline_executions (
    id BIGSERIAL PRIMARY KEY,
    context_key VARCHAR(100),
    query_hash VARCHAR(64) NOT NULL,
    model_used VARCHAR(50) NOT NULL,
    facts_used BIGINT[] NOT NULL,
    quality_score REAL NOT NULL,
    attempts INT NOT NULL,
    total_tokens INT NOT NULL,
    duration_ms INT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

---

## Summary

| Component | Owner | Responsibility |
|-----------|-------|---------------|
| `PipelineState`, `PipelineResult` | learn | Data types for pipeline execution |
| `AdaptationPipeline` | learn | Pipeline executor (controls flow) |
| `ContextSelector` protocol | learn | Contract for context selection |
| `QualityEvaluator` protocol | learn | Contract for quality evaluation |
| `ModelRouter` protocol | learn | Contract for model selection |
| `RetryStrategy` protocol | learn | Contract for retry decisions |
| `HybridSelector` | agent | Context selection implementation |
| `LLMJudgeEvaluator` | agent | Quality evaluation implementation |
| `ConfidenceBasedRouter` | agent | Model routing implementation |
| `ExponentialBackoffRetry` | agent | Retry strategy implementation |

**Learn provides the framework. Agent provides the intelligence.**
