# Goals

## Purpose

A framework for collecting, storing, and utilizing feedback data to customize LLM behavior. The
focus is on data infrastructure and training pipelines that support various customization
techniques.

## Core Thesis

**Data is the asset. Model weights are ephemeral.**

Store feedback data in model-agnostic formats. When better base models emerge, retrain on the same
data. Investment is in data collection infrastructure, not in any specific model's weights.

This means:
- Collect everything in portable formats (JSONL, PostgreSQL)
- Don't couple to specific model architectures
- Treat models as replaceable; treat data as permanent

## Design Principles

1. **Data-Centric** - Store all feedback in model-agnostic formats
2. **Layered Complexity** - Build from simple (context injection) to complex (fine-tuning)
3. **Feedback-Driven** - Every interaction is a potential learning signal
4. **Verifiable** - Track predictions against outcomes for calibration
5. **Privacy-First** - Secure handling of all data

## What This Framework Provides

### Data Collection
- Feedback storage (positive/negative signals, preference pairs)
- Interaction logging (engagement metrics)
- Prediction tracking (hypotheses with outcome verification)

### Training Infrastructure
- Export pipelines for classifier training
- Preference pair formatting for DPO/KTO
- Embedding generation and storage

### Customization Techniques
- RAG with context injection (immediate, no training)
- Classifier-based scoring (fast, cheap)
- LoRA adapters with preference optimization (deeper customization)

See [Learning Methods](02-learning-methods.md) for comprehensive technique reference.

## What This Is NOT

- **Not a foundation model** - Uses existing LLMs, doesn't train from scratch
- **Not a general ML platform** - Focused specifically on LLM customization
- **Not real-time learning** - Batch training, not online learning
- **Not unsupervised** - Model updates are driven by explicit signals

## Success Metrics

| Metric | Description |
|--------|-------------|
| Feedback volume | Amount of preference data collected |
| Export quality | Clean, usable training data |
| Model improvement | Measurable gains from customization |
| Calibration | Prediction accuracy matches confidence |
