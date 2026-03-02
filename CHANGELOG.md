# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Training stability detection: detects NaN gradients, loss spikes, and divergence
- Stability warnings in completed manifests (`unstable`, `stability_warnings` fields)
- Training parameter reproducibility: all effective params now persisted via `TRAINING_CONFIG_KEYS`
- Model-size-aware LoRA profiles with automatic detection (small/medium/large/xlarge)
- Gradient clipping support (`max_grad_norm`) in training config
- `--lora-profile` CLI option for manual profile override
- Symlink from adapter directory to completed manifest for traceability
- `ProfileDetectionError` exception for explicit handling of detection failures

### Changed
- `create_server()` now requires `lg: Logger` as first parameter
- Adapter version IDs now use full MD5 hash instead of truncated

### Fixed
- Add `readme` field to pyproject.toml so PyPI displays the README
- Update `create_server` docstring and architecture docs with new signature
- Add `max_grad_norm` to `_TRAINING_KEYS` for flat config override support

## [0.1.0] - 2026-02-25

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
- Export to DPO format (`dpo.export_preferences`) for TRL DPOTrainer
- Export to SFT format (`export_feedback_sft`) for supervised fine-tuning
- Export to classifier format (`export_feedback_classifier`) for binary classification
- LoRA training with QLoRA support (`train_lora`)
- DPO training (`train_dpo`)
- `lora.Config` and `RunConfig` for training configuration
- `AdapterRegistry` for managing trained adapters

#### Infrastructure
- PostgreSQL storage with pgvector extension for embeddings
- Alembic migrations for schema management
- GitHub Actions CI/CD pipeline
- Comprehensive test suite (unit, integration, e2e)

#### Documentation
- Example scripts for common workflows
- API reference documentation in README

[Unreleased]: https://github.com/serendip-ml/llm-kelt/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/serendip-ml/llm-kelt/releases/tag/v0.1.0
