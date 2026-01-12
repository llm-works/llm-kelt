# llm-learn

A framework for collecting and managing context that can be injected into LLM prompts.

## Features

- **Facts Storage** - Store and retrieve facts that get injected into system prompts
- **Feedback Collection** - Record explicit feedback signals (positive/negative/dismiss)
- **Preference Pairs** - Store chosen vs rejected responses for DPO training data
- **Interaction Tracking** - Record implicit signals (view, click, read, scroll)
- **Context Injection** - Build system prompts with relevant facts automatically included

## Installation

```bash
git clone https://github.com/serendip-ml/llm-learn.git
cd llm-learn
pip install -e .
```

## Quick Start

```python
from llm_learn import LearnClient

# Create client scoped to a profile
learn = LearnClient(profile_id=1)

# Add facts
learn.facts.add("Preferred language: Python", category="preferences")
learn.facts.add("Timezone: UTC", category="settings")

# Record feedback
learn.feedback.record(
    content_text="Some response text",
    signal="positive",
    strength=0.9,
)

# Record preference pair (for training data)
learn.preferences.record(
    context="Summarize this document",
    chosen="Concise 2-sentence summary",
    rejected="Verbose 10-paragraph summary",
)
```

## Context Injection

```python
from llm_learn.inference.context import ContextBuilder

# Build system prompt with facts injected
builder = ContextBuilder(learn.facts)
system_prompt = builder.build_system_prompt(
    base_prompt="You are a helpful assistant.",
    categories=["preferences"],  # Optional: filter by category
    min_confidence=0.5,          # Optional: filter by confidence
)
# Result includes facts formatted as "## About the user:" section
```

## Architecture

```
Profile
  ├── Facts (injected into prompts)
  ├── Feedback (explicit signals)
  ├── Preferences (DPO training pairs)
  ├── Interactions (implicit signals)
  ├── Content (deduplicated storage)
  ├── Directives (goals/rules)
  └── Predictions (hypothesis tracking)
```

All data is scoped to profiles within workspaces for multi-tenant isolation.

## Requirements

- Python 3.11+
- PostgreSQL 16+ (with pgvector extension)

## License

Apache 2.0
