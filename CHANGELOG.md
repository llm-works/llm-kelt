# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

#### Core Data Collection
- Facts storage and retrieval with categories, sources, and confidence scores
- Feedback collection (positive/negative/dismiss signals with strength)
- Preference pairs for DPO training data (chosen/rejected responses)
- Interaction tracking (view, click, read, scroll events)
- Content storage with deduplication
- Directives management (standing, one-time, rules)
- Predictions tracking with resolution and calibration
- Context-scoped data isolation

#### Inference
- `ContextBuilder` for system prompt augmentation with facts
- `ContextQuery` for high-level context-aware LLM interactions
- Multi-backend LLM client (Anthropic, OpenAI, OpenAI-compatible APIs)
- `Embedder` for generating embeddings via OpenAI-compatible API
- RAG support with semantic fact retrieval using `RAGArgs`
- `embed_missing_facts` utility for batch embedding
- Similarity search with category filtering (SQL-level)

#### Training
- Export to DPO format (`export_preferences_dpo`) for TRL DPOTrainer
- Export to SFT format (`export_feedback_sft`) for supervised fine-tuning
- Export to classifier format (`export_feedback_classifier`) for binary classification
- LoRA training with QLoRA support (`train_lora`)
- DPO training (`train_dpo`)
- `LoraConfig` and `TrainingConfig` for training configuration
- `AdapterRegistry` for managing trained adapters

#### Infrastructure
- PostgreSQL storage with pgvector extension for embeddings
- Alembic migrations for schema management
- GitHub Actions CI/CD pipeline
- Comprehensive test suite (unit, integration, e2e)

#### Documentation
- Example scripts for common workflows
- API reference documentation in README
