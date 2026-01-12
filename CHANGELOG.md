# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Facts storage and retrieval with categories, sources, and confidence scores
- Context injection via `ContextBuilder` for system prompt augmentation
- Feedback collection (positive/negative/dismiss signals)
- Preference pairs for DPO training data
- Interaction tracking (view, click, read, scroll)
- Content storage with deduplication
- Directives management (standing, one-time, rules)
- Predictions tracking with resolution and calibration
- Multi-backend LLM client (Anthropic, OpenAI, OpenAI-compatible)
- Profile-scoped data isolation
- PostgreSQL storage with pgvector support
- Alembic migrations
