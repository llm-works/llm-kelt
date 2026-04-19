# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `AsyncCompactor` base class for I/O-bound compaction strategies (e.g., LLM summarization)
- `Conversation.append_async()` for non-blocking compaction in async contexts
- Warns at construction when `AsyncCompactor` is used (reminds to use `append_async()`)
- Raises `RuntimeError` if sync `append()` triggers async compaction
- `Client.embedding_store` public property for entity-type agnostic vector storage
  (enables custom embeddings for non-fact entities like queries or documents)
- `EmbeddingStore` re-exported from `llm_kelt` for type hints
- `DeleteResult` dataclass for atomic fact deletion results (`.deleted`, `.not_found`, `.count`)
- `EmbeddingAdapter.delete_orphans(dry_run=False)` to clean up embeddings for deleted facts
- `kelt atomic vacuum [--dry-run]` CLI command for orphan embedding cleanup
- `llm_kelt.ensure_schema(lg, pg, schema_name=None)` top-level helper for
  embedding kelt into a foreign database. Provides a public migration path for
  consumers who have a `PG` instance but don't want to construct a full `Client`
  just to run migrations, without reaching into `core.*` modules. Shares its
  migration-setup sequence with `Client(ensure_schema=True)` via
  `Database.ensure_schema()` so both entry points run the same steps.
- Re-export `SchemaStatus` and `SchemaState` from `llm_kelt` so callers can
  inspect the result of `ensure_schema()` without importing from `core.*`.
- Conversation layer with stateful dialogue management (`llm_kelt.conversation`)
  - `Conversation` class with token tracking and automatic compaction
  - `Message`, `ToolCall`, and `Role` types (FieldDict-based, zero serialization overhead)
  - Sliding window and LLM-summarizing compaction strategies
  - File-based and PostgreSQL session storage backends
  - `kelt session` CLI commands for listing, showing, and deleting sessions
  - Example script (`examples/05_conversation.py`)
- Debug logging for conversation compaction (messages/tokens before/after, timing, usage ratio)

### Changed
- **Breaking**: Table `sessions` renamed to `conv_sessions` (avoids conflicts with agent tables)
- **Breaking**: Table `embeddings` renamed to `fact_embeddings` (avoids conflicts with agent tables)
- **Breaking**: `Conversation.__init__` now requires `lg: Logger` as the first parameter
- **Breaking**: `FactClient.delete()` now returns `DeleteResult` instead of `bool`, accepts
  `int | Iterable[int]`, and automatically cleans up associated embeddings
- `Conversation` moved from `llm_kelt.inference.query` to `llm_kelt.conversation.session`
  (re-exported from `llm_kelt.inference` for backward compatibility)
- `ContextQuery` now uses the new `Conversation` class with `messages_as_dicts()` for
  clean LLM API payloads

### Fixed
- Reduced flakiness in `tests/e2e/test_facts.py` by switching LLM calls to
  `temperature=0.0` and broadening the Python keyword assertion to accept any
  Python-ecosystem term (matches the existing pattern in the sibling category test).

### Documentation
- Add CLI reference (`docs/cli.md`) with full command documentation
- Expand README with prompt tuning, manifest-based training, and adapter registry examples
- Document training profiles table and stability detection features
- Add multi-schema operations example (`with_schema()`)
- Add conversation layer usage guide and CLI reference

## [0.2.0] - 2026-03-14

### Added
- DPO stacked adapters: VRAM-efficient reference model via adapter copying instead of loading a second model
- `kelt train merge` CLI tool for baking LoRA adapters into base model weights (required for VLM models where vLLM doesn't apply LoRA correctly)
- `extract_md5()` and `md5_matches()` utilities in storage module for flexible adapter version lookup (supports prefix..suffix notation)
- `Client.with_schema()` for per-operation schema selection without client caches
- `ScopedClient` for lazy-initializing schema-scoped operations
- `Database.scoped()` returning `ScopedDatabase` for session-level schema isolation
- `Source.schema_name` field for tracking training data provenance by schema
- `ManifestClient.get_manifest(md5)` for looking up manifests by adapter hash
- `kelt train deploy` CLI tool for deploying adapter versions
- `kelt train run --list-models` option to list available models
- `EmbeddingFilter` for flexible similarity search filtering with SQLAlchemy clause support
- Training stability detection: detects NaN gradients, loss spikes, and divergence
- Stability warnings in completed manifests (`unstable`, `stability_warnings` fields)
- Training parameter reproducibility: all effective params now persisted via `TRAINING_CONFIG_KEYS`
- Model-size-aware LoRA profiles with automatic detection (small/medium/large/xlarge)
- Gradient clipping support (`max_grad_norm`) in training config
- `--lora-profile` CLI option for manual profile override
- Symlink from adapter directory to completed manifest for traceability
- `ProfileDetectionError` exception for explicit handling of detection failures
- Prompt tuning as alternative PEFT method for large models (32B+) where LoRA can be unstable
- `use_rslora` parameter in LoRA config for rank-stabilized scaling (alpha/sqrt(r))
- `neftune_noise_alpha` in training config for embedding noise regularization

### Changed
- `create_server()` now requires `lg: Logger` as first parameter
- Adapter version IDs now use full MD5 hash instead of truncated
- Minimum TRL version bumped to 0.12 (required for conversational DPO format)
- **Breaking:** `ConfigurationError` renamed to `ConfigError`
- **Breaking:** `llm_kelt.core.exceptions` renamed to `llm_kelt.core.errors`
- **Breaking:** Removed `utc_now()` helper; use `datetime.now(UTC)` directly
- **Breaking:** Removed `reference_free` parameter from `train_dpo()` (TRL handles reference automatically)

### Fixed
- Add `readme` field to pyproject.toml so PyPI displays the README
- Update `create_server` docstring and architecture docs with new signature
- Add `max_grad_norm` to `_TRAINING_KEYS` for flat config override support
- SFT training now uses tokenizer's chat template for proper EOS token learning (errors if missing)
- DPO training data now uses chat message format for proper template handling
- BNB merge now preserves visual weights in index (fixes VLM model loading after merge)
- BNB merge falls back to standard path for non-sharded models instead of failing

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

[Unreleased]: https://github.com/llm-works/llm-kelt/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/llm-works/llm-kelt/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/llm-works/llm-kelt/releases/tag/v0.1.0
