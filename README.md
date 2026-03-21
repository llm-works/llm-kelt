# llm-kelt

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![Type Hints](https://img.shields.io/badge/type%20hints-100%25-brightgreen.svg)
[![Linting:
Ruff](https://img.shields.io/badge/linting-ruff-yellowgreen)](https://github.com/astral-sh/ruff)
[![CI](https://github.com/llm-works/llm-kelt/actions/workflows/ci.yml/badge.svg)](https://github.com/llm-works/llm-kelt/actions/workflows/ci.yml)
![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)

A framework for collecting, managing, and leveraging context for LLM applications. Supports fact
storage, feedback collection, preference pairs, RAG-based retrieval, and fine-tuning workflows.

## Features

- **Facts & Context Injection** - Store facts that get injected into LLM system prompts
- **RAG Retrieval** - Semantic search for relevant facts using embeddings
- **Feedback Collection** - Record explicit signals (positive/negative/dismiss)
- **Preference Pairs** - Store chosen vs rejected responses for DPO training
- **Training Export** - Export to DPO, SFT, and classifier formats
- **LoRA Fine-Tuning** - Train adapters with QLoRA support
- **Multi-Tenant** - Context-scoped data isolation

## Installation

```bash
# Basic installation
pip install llm-kelt

# With training dependencies (PyTorch, transformers, PEFT, TRL)
pip install llm-kelt[training]

# Development installation
git clone https://github.com/llm-works/llm-kelt.git
cd llm-kelt
pip install -e ".[dev]"
```

## Quick Start

### Setup

```python
from llm_kelt import Client

# Create client scoped to a context
kelt = Client(context_key="default")
kelt.migrate()  # Create database tables

# Add facts about the user
kelt.facts.add("Prefers concise explanations", category="preferences")
kelt.facts.add("Expert Python developer", category="background")
kelt.facts.add("Always include code examples", category="rules")
```

### Context Injection

```python
from llm_kelt.inference import ContextBuilder

# Build system prompt with facts injected
builder = ContextBuilder(kelt.facts)
system_prompt = builder.build_system_prompt(
    base_prompt="You are a helpful assistant.",
    categories=["preferences", "rules"],  # Optional: filter by category
)
# Result: "You are a helpful assistant.\n\n## About the user:\n### Preferences\n- ..."
```

### RAG-Based Retrieval

RAG (Retrieval-Augmented Generation) finds facts relevant to each query using semantic similarity.

```python
from llm_infer.client import LLMClient
from llm_kelt.inference import (
    ContextBuilder, ContextQuery, Embedder, RAGArgs, embed_missing_facts
)

# 1. Embed facts for semantic search (model name discovered from server)
embedder = Embedder(base_url="http://localhost:8001/v1")
await embed_missing_facts(logger, embedder, kelt.facts)

# 2. Create context-aware query interface
llm_client = LLMClient.from_config(config)
query = ContextQuery(
    client=llm_client,
    context_builder=ContextBuilder(kelt.facts),
    base_system_prompt="You are a helpful assistant.",
    embedder=embedder,
)

# 3. Ask questions - RAG finds relevant facts automatically
response = await query.ask(
    "What's my preferred coding style?",
    rag=RAGArgs(top_k=5, min_similarity=0.3),
)

# Filter by category
response = await query.ask(
    "What rules should I follow?",
    rag=RAGArgs(top_k=5, categories=["rules"]),
)

# Clean up
await embedder.close_async()
```

### Training Data Export

```python
from llm_kelt.training import export_feedback_sft
from llm_kelt.training.dpo import export_preferences

# Record preference pairs
kelt.atomic.preferences.record(
    context="Explain gradient descent",
    chosen="Concise, accurate explanation",
    rejected="Verbose, rambling explanation",
)

# Export to DPO format for TRL
result = export_preferences(
    session_factory=kelt.database.session,
    context_key=kelt.context_key,
    output_path="preferences.jsonl",
)
# Output: {"prompt": str, "chosen": str, "rejected": str}

# Export feedback to SFT format
result = export_feedback_sft(
    session_factory=kelt.database.session,
    context_key=kelt.context_key,
    output_path="feedback_sft.jsonl",
    signal="positive",
)
# Output: {"instruction": str, "output": str}
```

### LoRA Fine-Tuning

```python
from appinfra.log import LogConfig, LoggerFactory
from llm_kelt.training import train_lora
from llm_kelt.training.lora import Config as LoraConfig

lg = LoggerFactory.create_root(LogConfig.from_params(level="info"))

# Train LoRA adapter (requires pip install llm-kelt[training])
result = train_lora(
    lg=lg,
    data_path="feedback_sft.jsonl",
    output_dir="./my_adapter",
    base_model="Qwen/Qwen2.5-7B-Instruct",
    lora_config=LoraConfig(
        r=16,
        lora_alpha=32,
        use_rslora=True,  # Rank-stabilized scaling (alpha/sqrt(r))
    ),
    training_config={
        "num_epochs": 3,
        "batch_size": 4,
        "learning_rate": 2e-4,
        "max_grad_norm": 1.0,  # Gradient clipping
        "neftune_noise_alpha": 5.0,  # Embedding noise regularization
    },
    quantize=True,  # QLoRA for lower VRAM
)

print(f"Adapter saved to: {result.adapter.path}")
print(f"Train loss: {result.metrics['train_loss']:.4f}")
```

### Prompt Tuning

Alternative to LoRA for large models (32B+) where LoRA can be unstable:

```python
from llm_kelt.training.prompt import Config as PromptConfig
from llm_kelt.training.prompt import Trainer

# Configure soft prompt
prompt_config = PromptConfig(
    num_virtual_tokens=8,
    prompt_tuning_init="TEXT",
    prompt_tuning_init_text="You are a helpful assistant.",
)

# Train via manifest or directly
trainer = Trainer(lg, base_model, prompt_config, training_config)
result = trainer.train(data_path, output_dir)
```

### Manifest-Based Training

File-based workflow for reproducible training runs:

```yaml
# manifests/my-adapter.yaml
adapter: my-adapter
method: sft  # or dpo

source:
  schema_name: production  # Data source schema

data:
  format: inline  # or path
  records:
    - instruction: "What is 2+2?"
      output: "The answer is 4."

training:
  num_epochs: 3
  batch_size: 4
  learning_rate: 0.0002
```

```bash
# Run via CLI
kelt train run manifests/my-adapter.yaml

# Or programmatically
from llm_kelt.training import Runner
from llm_kelt.training.storage import FileStorage

storage = FileStorage(lg, registry_path)
runner = Runner(lg, storage, model_locations=[Path("~/models")])
result = runner.run(manifest_path)
```

### Training Profiles

LoRA profiles are auto-detected based on model size:

| Profile | Model Size | LoRA Rank | Alpha | Batch |
|---------|-----------|-----------|-------|-------|
| small | <3B | 8 | 16 | 8 |
| medium | 3-14B | 16 | 32 | 4 |
| large | 14-32B | 32 | 64 | 2 |
| xlarge | >32B | 64 | 128 | 1 |

Override with `--lora-profile` in CLI or via config.

Training includes stability detection for NaN gradients, loss spikes, and divergence. Warnings are
recorded in completed manifests.

### Adapter Registry

Manage trained adapters with versioning and deployment:

```python
from llm_kelt.training import AdapterRegistry
from llm_kelt.training.storage import FileStorage

storage = FileStorage(lg, base_path="~/adapters")
registry = AdapterRegistry(lg, storage, infer_url="http://localhost:8000")

# Register after training
info = registry.register_and_refresh(
    training_result=result,
    key="my-adapter",
    description="Fine-tuned on customer data",
    deploy=True,  # Make available for inference
)

# List adapters and versions
adapters = registry.list()
versions = storage.list_versions("my-adapter")

# Deploy specific version
storage.deploy_adapter("my-adapter", version_id, policy="replace")

# Refresh inference server
registry.refresh("my-adapter")
```

### Multi-Schema Operations

Use `with_schema()` for per-operation schema selection:

```python
from llm_kelt import ClientFactory, ClientContext

# Schema-agnostic client
context = ClientContext(context_key="my-agent")
client = factory.create_from_config(context=context, config=config)

# Schema specified at operation time
client.with_schema("production").atomic.facts.add("User prefers concise responses")
client.with_schema("staging").atomic.preferences.record(...)

# Useful for training pipelines reading from multiple schemas
schema = manifest.source.schema_name
data = client.with_schema(schema).atomic.preferences.list()
```

## Architecture

```
Isolation Context (context_key)
  ├── Facts           → Injected into prompts (with embeddings for RAG)
  ├── Feedback        → Explicit signals (positive/negative)
  ├── Preferences     → DPO training pairs (chosen/rejected)
  ├── Interactions    → Implicit signals (view, click, scroll)
  ├── Content         → Deduplicated content storage
        ├── Directives      → Goals and rules
        └── Predictions     → Hypothesis tracking
```

### Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                           COLLECTION                                 │
│  Facts  │  Feedback  │  Preferences  │  Interactions  │  Content    │
└────────────────────────────┬────────────────────────────────────────┘
                             │
         ┌───────────────────┼───────────────────┐
         ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│    INFERENCE    │ │    TRAINING     │ │    ANALYSIS     │
│                 │ │                 │ │                 │
│ • Context       │ │ • Export DPO    │ │ • Stats         │
│   Injection     │ │ • Export SFT    │ │ • Trends        │
│ • RAG Retrieval │ │ • LoRA Training │ │ • Insights      │
│ • Embeddings    │ │ • DPO Training  │ │                 │
└─────────────────┘ └─────────────────┘ └─────────────────┘
```

## Examples

See the [`examples/`](examples/) directory for complete working examples:

- [`01_facts_and_context.py`](examples/01_facts_and_context.py) - Facts storage and context
  injection
- [`02_rag_retrieval.py`](examples/02_rag_retrieval.py) - RAG with semantic search
- [`03_training_export.py`](examples/03_training_export.py) - Export to training formats
- [`04_lora_training.py`](examples/04_lora_training.py) - LoRA fine-tuning workflow

## API Reference

### Core

| Class | Description |
|-------|-------------|
| `Client` | Main entry point, scoped to a context |
| `Client.with_schema()` | Per-operation schema selection |
| `ScopedClient` | Lazy-initializing schema-scoped client |
| `FactsClient` | Store and retrieve facts |
| `FeedbackClient` | Record explicit feedback signals |
| `PreferencesClient` | Store preference pairs |

### Inference

| Class/Function | Description |
|----------------|-------------|
| `ContextBuilder` | Build system prompts with injected facts |
| `ContextQuery` | High-level context-aware query interface |
| `Embedder` | Generate embeddings via OpenAI-compatible API |
| `RAGArgs` | Configuration for RAG retrieval |
| `embed_missing_facts` | Batch embed facts without embeddings |

### Training

| Class/Function | Description |
|----------------|-------------|
| `dpo.export_preferences` | Export preference pairs for DPO |
| `export_feedback_sft` | Export feedback for SFT |
| `export_feedback_classifier` | Export for binary classification |
| `train_lora` | Train LoRA adapter with SFT |
| `train_dpo` | Train with Direct Preference Optimization |
| `lora.Config` | LoRA hyperparameters (`r`, `lora_alpha`, `use_rslora`) |
| `prompt.Config` | Prompt tuning config (`num_virtual_tokens`, init settings) |
| `AdapterRegistry` | Manage trained adapters with versioning |
| `FileStorage` | File-based adapter storage backend |
| `Runner` | Execute training from manifest files |
| `Manifest` | Training manifest schema |
| `build_training_config` | Build config from profile with overrides |

### CLI

See [CLI Reference](docs/cli.md) for full documentation.

| Command | Description |
|---------|-------------|
| `kelt train run` | Run training from manifest |
| `kelt train sft` | Direct SFT training |
| `kelt train dpo` | Direct DPO training |
| `kelt train adapters` | List registered adapters |
| `kelt train deploy` | Deploy adapter version |
| `kelt train merge` | Merge LoRA into base model |

## Requirements

- Python 3.11+
- PostgreSQL 16+ with pgvector extension
- For training: CUDA GPU (or MPS on Apple Silicon)

## Configuration

1. Copy the environment template and customize paths:

```bash
cp .env.yaml.example .env.yaml
```

Edit `.env.yaml` with your local paths:

```yaml
paths:
  models: !path ~/models/huggingface    # HuggingFace models directory
  adapters: !path ~/models/adapters     # Trained LoRA adapters
```

2. The main config is in `etc/llm-kelt.yaml`. Database and LLM settings:

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
```

## License

Apache 2.0
