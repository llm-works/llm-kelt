# Quickstart

Five minutes from install to a working RAG query.

## 1. Postgres

The library needs Postgres 16+ with pgvector. Three paths:

**A. `appinfra pg up`.** No repo checkout required. `appinfra` is a runtime dep, so the CLI
lands on `PATH` after `pip install llm-kelt`. Starts pgvector:pg18 as container
`llm-works-pg` on port 25432 with trust auth; podman preferred over docker. Matches the
default URL that `python -m llm_kelt.examples.quickstart` connects to, so the smoke works
with no extra config. Stop with `appinfra pg down`. Requires `appinfra>=0.11.0`.

**B. Cloned repo — `make pg.server.up`.** Same container and defaults as (A), via the
shipped `etc/pg.yaml`. Works with docker or podman (`INFRA_CONTAINER_CMD` in
`Makefile.local` selects the runtime). Stop with `make pg.server.down`.

**C. Existing Postgres.** Any Postgres 16+ with the `vector` extension installable
(`CREATE EXTENSION vector`) works. Point the URL in step 3 at it.

## 2. Install

```bash
pip install llm-kelt
```

## 3. Config

kelt ships a packaged base config (`llm_kelt/etc/llm-kelt.yaml`) with defaults matching
paths A and B — port 25432, trust auth, database `kelt` (auto-created). No local file
needed for those paths.

To override any setting — path C's URL, credentials, ports, adapter paths — drop a user
overlay at `~/.config/llm-works/llm-kelt.yaml`. The v1 config protocol picks it up ahead
of the packaged base:

```yaml
dbs:
  main:
    url: postgresql://user:pass@your-host:5432/your-db
    extensions: [vector]
```

The `llm`, `embedding`, and `kelt.adapters` sections are added to the overlay as those
subsystems are enabled (section 6 below, and the [Context & RAG](context-and-rag.md) /
[Training](training.md) tutorials).

## 4. First client

```python
from appinfra.config import Config
from appinfra.log import create_root_lg
from llm_kelt import ClientContext, ClientFactory

config = Config.from_spec("llm-works", "llm-kelt")
lg = create_root_lg(level="warning")
kelt = ClientFactory(lg).create_from_config(
    context=ClientContext(context_key="quickstart"),
    config=config,
)
```

`Config.from_spec` walks the v1 precedence chain: user overlay (if any), then the
packaged base. On first run this creates the `public` schema tables. The `context_key`
scopes every read and write — nothing you record here is visible to a client using a
different key.

## 5. Record and read a fact

```python
fact_id = kelt.atomic.assertions.add(
    "Timezone: UTC",
    category="settings",
)

for fact in kelt.atomic.assertions.list_active():
    print(fact.id, fact.category, fact.content)
```

`add()` returns the fact ID as `int`. `list_active()` returns `list[Fact]` — ORM objects with
`.id`, `.content`, `.category`, `.confidence`, `.source`, `.created_at`, `.active`.

## 6. Inject facts into a system prompt

```python
from llm_kelt.inference import ContextBuilder

kelt.atomic.assertions.add("Prefers concise, code-first answers", category="style")
kelt.atomic.assertions.add("Use type hints on every function", category="style")

builder = ContextBuilder(kelt.atomic.assertions)
system_prompt = builder.build_system_prompt("You are a helpful assistant.")
print(system_prompt)
```

Output:

```text
You are a helpful assistant.

## About the user:

### settings
- Timezone: UTC

### style
- Prefers concise, code-first answers
- Use type hints on every function
```

Filter by category or cap total facts:

```python
builder.build_system_prompt("...", categories=["style"], max_facts=20)
```

## 7. Ask a question with context

Add an LLM section to your config:

```yaml
llm:
  default: local
  backends:
    local:
      base_url: http://localhost:8000/v1
      model: default
```

Then:

```python
import asyncio
from llm_infer.client import Factory as LLMFactory
from llm_kelt.inference import ContextQuery

llm_client = LLMFactory(lg).from_config(config.llm.to_dict())

query = ContextQuery(
    client=llm_client,
    context_builder=builder,
    base_system_prompt="You are a helpful assistant.",
)


async def main():
    answer = await query.ask("How should I structure my next function?")
    print(answer)
    await llm_client.aclose()


asyncio.run(main())
```

The facts get injected into the system prompt on every call. If you don't want them for a
specific call, use `ask_without_facts(...)`.

## 8. Turn it into RAG

Static injection sends every fact every time. RAG sends only the facts relevant to the current
question. Add an embedding backend to the config:

```yaml
embedding:
  type: openai
  base_url: http://localhost:8001/v1
  model: text-embedding-3-small
```

Then embed the facts once and enable RAG on the query:

```python
from llm_kelt.inference import RAGArgs, embed_missing_facts

embedder = LLMFactory(lg).embeddings_from_config(config.embedding.to_dict())


async def main():
    await embed_missing_facts(
        lg=lg,
        embedder=embedder,
        embedding_adapter=kelt.atomic.embeddings,
        dimensions=384,
    )

    query = ContextQuery(
        client=llm_client,
        context_builder=builder,
        base_system_prompt="You are a helpful assistant.",
        embedder=embedder,
        embedding_adapter=kelt.atomic.embeddings,
    )

    answer = await query.ask(
        "What tone should I use?",
        rag=RAGArgs(top_k=5, min_similarity=0.3),
    )
    print(answer)

    await embedder.aclose()
    await llm_client.aclose()


asyncio.run(main())
```

Only the facts semantically close to `"What tone should I use?"` reach the prompt.

## What next

- Record more than assertions: [atomic memory tutorial](atomic-memory.md).
- Multi-turn dialogues with token accounting: [conversation](conversation.md).
- Train an adapter from feedback and preferences: [training](training.md).
- Read [concepts](concepts.md) once to solidify the mental model.
