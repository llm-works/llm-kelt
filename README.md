# llm-kelt

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![Type Hints](https://img.shields.io/badge/type%20hints-100%25-brightgreen.svg)
[![Linting: Ruff](https://img.shields.io/badge/linting-ruff-yellowgreen)](https://github.com/astral-sh/ruff)
[![CI](https://github.com/llm-works/llm-kelt/actions/workflows/ci.yml/badge.svg)](https://github.com/llm-works/llm-kelt/actions/workflows/ci.yml)
![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)

Persistent memory and fine-tuning data for LLM applications, backed by Postgres.

Store the things an LLM needs to know or learn from — facts, feedback, preferences, predictions,
directives — under an isolation key. Retrieve them for prompt injection or RAG. Export them to
DPO/SFT/classifier datasets. Train LoRA/DPO/Prompt adapters from the exported data.

## Requirements

- Python 3.11+
- PostgreSQL 16+ with the `vector` extension (pgvector)
- For training: CUDA GPU (or MPS on Apple Silicon)

## Install

```bash
pip install llm-kelt              # runtime
pip install llm-kelt[training]    # + torch / transformers / peft / trl
```

## Minimal example

```python
from appinfra.config import Config
from appinfra.log import LogConfig, LoggerFactory
from llm_kelt import ClientContext, ClientFactory
from llm_kelt.inference import ContextBuilder

config = Config("etc/llm-kelt.yaml")
lg = LoggerFactory.create_root(LogConfig.from_params(level="warning"))

kelt = ClientFactory(lg).create_from_config(
    context=ClientContext(context_key="my-agent"),
    config=config,
)

kelt.atomic.assertions.add("Timezone: UTC", category="settings")
kelt.atomic.assertions.add("Prefers concise, code-first answers", category="style")

system_prompt = ContextBuilder(kelt.atomic.assertions).build_system_prompt(
    base_prompt="You are a helpful assistant.",
)
# → "You are a helpful assistant.\n\n## About the user:\n- Timezone: UTC\n- ..."
```

That's the whole shape: put things in under a `context_key`, pull them back out grouped for
prompt injection. Everything else (RAG, feedback, preferences, training) builds on the same
model.

## Where to go next

- [Quickstart](docs/quickstart.md) — 5 minutes from install to first RAG query.
- [Concepts](docs/concepts.md) — context keys, schemas, atomic vs KG. Read once before the tutorials.
- [Atomic memory](docs/atomic-memory.md) — the seven fact clients (assertions, feedback,
  preferences, predictions, directives, interactions, solutions) and how they relate.
- [Context & RAG](docs/context-and-rag.md) — embedding facts, semantic search, `ContextQuery`.
- [Conversation](docs/conversation.md) — multi-turn sessions, token accounting, compaction, storage.
- [Training](docs/training.md) — manifest workflow, LoRA/DPO/SFT/Prompt, exports, adapter registry.
- [Knowledge graph](docs/knowledge-graph.md) — entities, aliases, hierarchical scopes.
- [Multi-schema](docs/multi-schema.md) — `SchemaMode`, `with_schema()`, isolation.
- [CLI reference](docs/cli.md) — `kelt atomic|proxy|train|session`.
- [Glossary](docs/glossary.md) — project-specific terms.

## Configuration

The library reads its config from `etc/llm-kelt.yaml`. Key sections:

```yaml
dbs:
  main:
    url: postgresql://user:pass@localhost:5432/llm_kelt
    extensions: [vector]

llm:
  default_backend: local
  backends:
    local:
      base_url: http://localhost:8000/v1
      model: default

embedding:
  type: openai
  base_url: http://localhost:8001/v1
  model: text-embedding-3-small

adapters:
  lora:
    base_path: ~/models/adapters
```

`llm`, `embedding`, and `adapters` are only required for the subsystems that use them
(`ContextQuery`, RAG, and training respectively).

## Examples

Runnable scripts in [`examples/`](examples/):

- `01_facts_and_context.py` — assertions + `ContextBuilder`.
- `02_rag_retrieval.py` — embeddings, `search_similar`, `ContextQuery` with RAG.
- `03_training_export.py` — feedback + preferences → DPO/SFT/classifier JSONL.
- `04_lora_training.py` — end-to-end LoRA training.
- `05_conversation.py` — `Conversation`, compaction, `FileSessionStorage`.

## License

Apache 2.0
