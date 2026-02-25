# llm-kelt

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![Type Hints](https://img.shields.io/badge/type%20hints-100%25-brightgreen.svg)
[![Linting:
Ruff](https://img.shields.io/badge/linting-ruff-yellowgreen)](https://github.com/astral-sh/ruff)
[![CI](https://github.com/serendip-ml/llm-kelt/actions/workflows/ci.yml/badge.svg)](https://github.com/serendip-ml/llm-kelt/actions/workflows/ci.yml)
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
git clone https://github.com/serendip-ml/llm-kelt.git
cd llm-kelt
pip install -e ".[dev]"
```

## Quick Start

### Setup

```python
from llm_kelt import Client

# Create client scoped to a context
learn = Client(context_key="default")
learn.migrate()  # Create database tables

# Add facts about the user
learn.facts.add("Prefers concise explanations", category="preferences")
learn.facts.add("Expert Python developer", category="background")
learn.facts.add("Always include code examples", category="rules")
```

### Context Injection

```python
from llm_kelt.inference import ContextBuilder

# Build system prompt with facts injected
builder = ContextBuilder(learn.facts)
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
await embed_missing_facts(logger, embedder, learn.facts)

# 2. Create context-aware query interface
llm_client = LLMClient.from_config(config)
query = ContextQuery(
    client=llm_client,
    context_builder=ContextBuilder(learn.facts),
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
learn.atomic.preferences.record(
    context="Explain gradient descent",
    chosen="Concise, accurate explanation",
    rejected="Verbose, rambling explanation",
)

# Export to DPO format for TRL
result = export_preferences(
    session_factory=learn.database.session,
    context_key=learn.context_key,
    output_path="preferences.jsonl",
)
# Output: {"prompt": str, "chosen": str, "rejected": str}

# Export feedback to SFT format
result = export_feedback_sft(
    session_factory=learn.database.session,
    context_key=learn.context_key,
    output_path="feedback_sft.jsonl",
    signal="positive",
)
# Output: {"instruction": str, "output": str}
```

### LoRA Fine-Tuning

```python
from appinfra.log import LogConfig, LoggerFactory
from llm_kelt.training import train_lora, RunConfig
from llm_kelt.training.lora import Config as LoraConfig

lg = LoggerFactory.create_root(LogConfig.from_params(level="info"))

# Train LoRA adapter (requires pip install llm-kelt[training])
result = train_lora(
    lg=lg,
    data_path="feedback_sft.jsonl",
    output_dir="./my_adapter",
    base_model="Qwen/Qwen2.5-7B-Instruct",
    lora_config=LoraConfig(r=16, lora_alpha=32),
    training_config=RunConfig(
        num_epochs=3,
        batch_size=4,
        learning_rate=2e-4,
    ),
    quantize=True,  # QLoRA for lower VRAM
)

print(f"Adapter saved to: {result.adapter_path}")
print(f"Train loss: {result.metrics['train_loss']:.4f}")
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
| `lora.Config` | LoRA hyperparameters |
| `RunConfig` | Training hyperparameters |
| `AdapterRegistry` | Manage trained adapters |

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
