# Context injection and RAG

Getting stored facts into an LLM call. Two strategies:

- **Static injection** — include facts in the system prompt on every call (up to `max_facts`,
  default 100). Zero infrastructure (no embeddings, no vector DB queries), cheap to reason
  about, but the prompt grows with the fact set.
- **RAG** — embed the facts once, embed the question at call time, include only the top-k
  facts by cosine similarity. Prompt size bounded by `top_k`, not memory size.

Both use the same `ContextBuilder` and `ContextQuery` — RAG is an argument to `.ask()`.

## Setup

```python
from appinfra.config import Config
from appinfra.log import LogConfig, LoggerFactory
from llm_kelt import ClientContext, ClientFactory
from llm_kelt.inference import (
    ContextBuilder,
    ContextQuery,
    RAGArgs,
    embed_missing_facts,
)

config = Config("etc/llm-kelt.yaml")
lg = LoggerFactory.create_root(LogConfig.from_params(level="warning"))
kelt = ClientFactory(lg).create_from_config(
    context=ClientContext(context_key="tutorial:rag"),
    config=config,
)
```

## Static injection with `ContextBuilder`

```python
class ContextBuilder:
    def __init__(self, facts_client: AssertionsClient)

    def build_system_prompt(
        self,
        base_prompt: str = "",
        categories: list[str] | None = None,
        min_confidence: float = 0.0,
        max_facts: int = 100,
        fact_position: Literal["append", "prepend"] = "append",
    ) -> str

    def build_system_prompt_from_facts(
        self,
        base_prompt: str,
        facts: list[Fact],
        fact_position: Literal["append", "prepend"] = "append",
    ) -> str

    def get_facts_summary(self, max_facts: int = 100) -> dict
```

Only assertions are injected — feedback, preferences, etc. are structured data, not for the
prompt. The builder groups facts by category when any have one, or produces a flat list
otherwise.

```python
kelt.atomic.assertions.add("Timezone: UTC", category="settings")
kelt.atomic.assertions.add("Prefers concise, code-first answers", category="style")
kelt.atomic.assertions.add("Use type hints on every function", category="style")

builder = ContextBuilder(kelt.atomic.assertions)
prompt = builder.build_system_prompt("You are a helpful assistant.")
```

Produces:

```text
You are a helpful assistant.

## About the user:

### settings
- Timezone: UTC

### style
- Prefers concise, code-first answers
- Use type hints on every function
```

Common filters:

```python
# Only style facts, at most 20 of them
builder.build_system_prompt("...", categories=["style"], max_facts=20)

# Only high-confidence facts
builder.build_system_prompt("...", min_confidence=0.8)

# Prepend facts before the base instruction instead of after
builder.build_system_prompt("...", fact_position="prepend")
```

If you already have a list of `Fact` objects from somewhere else (e.g. a similarity search
result you're going to inject), use the `_from_facts` variant to bypass the query:

```python
scored = kelt.atomic.embeddings.search_similar(query_vec, model="…", top_k=5)
picked = [sf.entity for sf in scored]
prompt = builder.build_system_prompt_from_facts("You are an assistant.", picked)
```

## Wire up an LLM client

```python
from llm_infer.client import Factory as LLMFactory

llm_factory = LLMFactory(lg)
llm_client = llm_factory.from_config(config.llm.to_dict())
```

The LLM client is an `llm_infer.client.ChatClient` — any object with a compatible
`chat_async(messages=..., system=..., **kwargs)` interface works. `llm-infer` provides
adapters for local (OpenAI-compatible) and hosted (OpenAI, Anthropic) backends.

## `ContextQuery` — one-shot ask with injected context

```python
class ContextQuery:
    def __init__(
        self,
        client: ChatClient,
        context_builder: ContextBuilder,
        base_system_prompt: str = "",
        temperature: float = 0.7,
        embedder: EmbeddingClient | None = None,
        embedding_adapter: EmbeddingAdapter | None = None,
    )

    async def ask(
        self,
        question: str,
        conversation: Conversation | None = None,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        include_facts: bool = True,
        fact_categories: list[str] | None = None,
        rag: RAGArgs | None = None,
        *,
        embedder_context: dict | None = None,
    ) -> str

    async def ask_without_facts(self, question, conversation=None, ...) -> str
    def get_injected_context(self, system_prompt=None, fact_categories=None) -> str
    async def close() -> None
```

Static injection example:

```python
import asyncio


async def demo():
    query = ContextQuery(
        client=llm_client,
        context_builder=builder,
        base_system_prompt="You are a helpful assistant.",
        temperature=0.3,
    )
    answer = await query.ask("How should I structure my next function?")
    print(answer)
    await query.close()


asyncio.run(demo())
```

`query.close()` closes the underlying LLM client. It's also an async context manager, so:

```python
async with ContextQuery(
    client=llm_client, context_builder=builder, base_system_prompt="..."
) as query:
    print(await query.ask("..."))
```

## RAG

Adding RAG to the same query is two changes: embed the facts, and pass `rag=` on `ask()`.

### Step 1 — set up an embedder

```yaml
# etc/llm-kelt.yaml
embedding:
  type: openai
  base_url: http://localhost:8001/v1
  model: text-embedding-3-small
```

```python
embedder = llm_factory.embeddings_from_config(config.embedding.to_dict())
```

`embedder.model` gives the model name, `embedder.embed_async(text) → EmbeddingResult` embeds
a single string.

### Step 2 — embed the facts

```python
async def embed_all():
    result = await embed_missing_facts(
        lg=lg,
        embedder=embedder,
        embedding_adapter=kelt.atomic.embeddings,
        dimensions=384,
        batch_size=50,
    )
    print(f"embedded {result.processed}, failed {result.failed}")
```

`embed_missing_facts` is idempotent: it lists facts that don't yet have an embedding for the
given `(model, dimensions)` pair, batches them, and writes results back. Re-run any time
after adding new assertions.

`dimensions` picks the quantization table (`f16_384`, `i8_768`, etc.). Match your embedder's
dimensionality. The library stores embeddings in the format configured on `EmbeddingConfig`
(`F16` by default — half-vec, 2× smaller than `F32`).

Pass an optional `context=` dict to attribute the embedder call to a cost/tracking label:

```python
await embed_missing_facts(
    lg=lg,
    embedder=embedder,
    embedding_adapter=kelt.atomic.embeddings,
    dimensions=384,
    context={"source": "nightly-embed"},
)
```

### Step 3 — query with RAG

```python
async def rag_demo():
    query = ContextQuery(
        client=llm_client,
        context_builder=builder,
        base_system_prompt="You are a helpful assistant.",
        embedder=embedder,
        embedding_adapter=kelt.atomic.embeddings,
    )
    answer = await query.ask(
        "What tone should I use in error messages?",
        rag=RAGArgs(top_k=5, min_similarity=0.3),
    )
    print(answer)
    await query.close()
```

On each `ask()` with `rag=` set, `ContextQuery`:

1. Embeds `question` with the configured embedder.
2. Calls `embedding_adapter.search_similar(query_vec, top_k, min_similarity, categories)`.
3. Injects the returned facts via `build_system_prompt_from_facts` instead of the static list.

### `RAGArgs`

```python
@dataclass
class RAGArgs:
    top_k: int = 10
    min_similarity: float = 0.3
    model: str | None = None  # override; defaults to embedder.model
    categories: list[str] | None = None
```

Category filter is applied server-side by the vector query — no post-filter.

## Just the similarity search

If you want the search result outside a query flow (dashboards, offline scoring):

```python
result = await embedder.embed_async("How do I handle timeouts?")
scored = kelt.atomic.embeddings.search_similar(
    query=result.embedding,
    model=embedder.model,
    top_k=10,
    min_similarity=0.4,
    categories=["errors", "retries"],
)
for sf in scored:
    print(f"{sf.score:.3f}  [{sf.entity.category}] {sf.entity.content}")
```

`scored` is `list[ScoredEntity[Fact]]` — each has `.score: float` and `.entity: Fact`.

## Batch retrieval — avoid N+1

Reading many facts by ID? Use the batch method (added in #83):

```python
ids = [12, 34, 56, 78]
by_id = kelt.atomic.embeddings.get_embeddings(ids, model="text-embedding-3-small")
# → dict[int, list[float]]
```

This is one SQL round-trip instead of one per fact.

## Categories vs RAG — when to use which

| Use | When |
|---|---|
| `categories=` on `build_system_prompt` | You know which category is relevant to this call. Cheap, no embedding needed. |
| `RAGArgs(categories=[...])` | You want similarity ranking *within* one or more categories. |
| `RAGArgs` alone | Category isn't known in advance — let similarity pick. |
| No `rag`, no categories | Small fact set. Just inject everything. |

## Quantization and dimensions

Embeddings are stored in a per-format, per-dimension table (e.g. `embeddings_f16_384`). The
default is `F16` (half-vec, 2× smaller than `F32`) at 384 dimensions.

To change the format globally, override `EmbeddingConfig` when constructing the `Client`
directly (not via `create_from_config`). See the
[`llm_kelt.embedding`](../llm_kelt/embedding/__init__.py)
package for `F32`, `F16`, `I8`, `I4` and their trade-offs.

## Full working script

See [`examples/02_rag_retrieval.py`](../examples/02_rag_retrieval.py) — populates 12 facts
across four categories, embeds them, runs three demos (raw similarity search, RAG vs static
prompt, full RAG query with LLM), and falls back to synthetic embeddings if no embedding
backend is running so the script still runs end-to-end.
