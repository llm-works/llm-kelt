# Framework Strategy

Which approaches the framework supports, and when to use them.

## Approach Selection

From the [learning methods](02-learning-methods.md), the framework supports:

| Approach | Use Case | Why |
|----------|----------|-----|
| **RAG** | Immediate context injection | No training needed, works with any LLM |
| **Classifiers** | Fast scoring/filtering | Cheap, fast inference |
| **Embeddings** | Similarity search | Required for RAG, enables clustering |
| **LoRA + DPO** | Behavior customization | Efficient fine-tuning for local models |

## What the Framework Does NOT Support

| Approach | Reason |
|----------|--------|
| **Full fine-tuning** | Too expensive, LoRA is sufficient |
| **RLHF** | Overkill, DPO achieves similar results more simply |
| **Training from scratch** | Uses existing base models (see [out of scope](02-learning-methods.md#out-of-scope)) |
| **Continued pre-training** | LoRA + RAG achieves similar results at fraction of cost |
| **Prompt tuning** | Not available via API, limited value |
| **Reward modeling** | Only needed for RLHF, which we're not doing |

## Future Considerations

Approaches that may be added as the framework matures:

| Approach | When to Consider | Trigger |
|----------|------------------|---------|
| **KTO** | If paired preferences are hard to collect | Have thumbs up/down but not A vs B |
| **Synthetic Data** | Bootstrapping training | Need more training examples than organic feedback provides |
| **RLAIF** | Scaling preference collection | Human feedback becomes bottleneck |
| **Knowledge Distillation** | Deployment cost reduction | Need faster/cheaper inference |
| **Active Learning** | Optimizing labeling effort | Limited annotation budget |
| **Continual Learning** | Ongoing model updates | Need to update without full retraining |

See [learning methods](02-learning-methods.md) for full details on each technique.

## Implementation Layers

### Layer 1: Context Injection

**Goal:** Immediate customization with no training.

- Embed content as it's ingested
- Retrieve relevant context at query time
- Inject into LLM prompt
- Works with API models (Claude, GPT) and local models

**Data needed:** Content with embeddings

### Layer 2: Scoring

**Goal:** Fast relevance/quality scoring.

- Train classifiers on feedback signals
- Fast inference for filtering
- Periodic retraining as feedback accumulates

**Data needed:** Labeled examples (100+)

### Layer 3: Behavior Customization

**Goal:** Model learns specific behaviors.

- LoRA adapters with DPO/KTO
- Periodic retraining
- Hot-swap into local inference

**Data needed:** 500+ preference pairs (chosen/rejected) or binary feedback

**Note:** Only applies when running local LLM. If API-only, use Layer 1 and 2.

---

## Trade-offs

### Simplicity over Power
- LoRA over full fine-tune
- DPO/KTO over RLHF (no reward model needed)
- Classifiers over complex LLM chains for scoring

### Portability over Lock-in
- Store everything in PostgreSQL (portable)
- JSONL exports for training (tool-agnostic)
- Support both API and local models

### Iteration over Perfection
- Start with RAG (immediate results)
- Add classifiers when feedback accumulates
- Add LoRA when preference pairs are available

---

## Resource Requirements

| Layer | GPU | Storage |
|-------|-----|---------|
| Layer 1 (Context) | None (embeddings via API or local) | PostgreSQL + pgvector |
| Layer 2 (Scoring) | Optional (CPU works for small classifiers) | + classifier artifacts |
| Layer 3 (Behavior) | 1x 24GB+ (RTX 4090 or better) | + LoRA adapters |

For LoRA training on larger models (70B+), need 2x 48GB or cloud GPU.

---

## Success Metrics

| Layer | Metric | Target |
|-------|--------|--------|
| Layer 1 | RAG retrieval precision | >70% relevant |
| Layer 2 | Classifier accuracy | >80% on held-out |
| Layer 3 | DPO win rate | >60% vs base |
