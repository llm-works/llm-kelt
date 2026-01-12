# Learning Methods

Comprehensive reference of techniques for LLM customization and adaptation. Each approach has
trade-offs; there is no single "best" approach.

## Overview Table

| Approach | What It Does | Data Required | Compute | Latency to Results |
|----------|--------------|---------------|---------|-------------------|
| [RAG](#rag) | Retrieve relevant context at inference | Embeddings | Low | Immediate |
| [Memory/Context](#memorycontext-injection) | Inject facts into prompt | Structured data | None | Immediate |
| [Classifiers](#classifiers) | Train small models for scoring | Labeled examples | Low | Hours |
| [Embeddings](#embedding-models) | Learn similarity representations | Pairs/triplets | Medium | Hours |
| [Prompt Tuning](#prompt-tuning) | Learn soft prompt tokens | Examples | Low | Hours |
| [LoRA/Adapters](#loraadapters) | Efficient parameter-efficient fine-tuning | Instructions or preferences | Medium | Hours-Days |
| [SFT](#sft-supervised-fine-tuning) | Supervised instruction tuning | (instruction, response) | Medium-High | Days |
| [DPO](#dpo-direct-preference-optimization) | Preference alignment without reward model | Chosen/rejected pairs | Medium | Days |
| [DPO Variants](#dpo-variants-orpo-kto-ipo) | Improved preference optimization | Chosen/rejected pairs | Medium | Days |
| [RLHF](#rlhf-reinforcement-learning-from-human-feedback) | Reward-based alignment | Comparisons + prompts | High | Days-Weeks |
| [RLAIF](#rlaif-reinforcement-learning-from-ai-feedback) | AI-generated feedback alignment | Prompts + AI judge | Medium-High | Days |
| [Reward Modeling](#reward-modeling) | Train scalar reward predictor | Comparisons | Medium | Hours-Days |
| [Knowledge Distillation](#knowledge-distillation) | Transfer knowledge to smaller model | Teacher outputs | Medium | Days |
| [Synthetic Data](#synthetic-data-generation) | Generate training data with LLMs | Seed examples | Low-Medium | Hours |
| [Active Learning](#active-learning) | Select most informative samples | Unlabeled pool | Low | Ongoing |
| [Continual Learning](#continual-learning) | Learn incrementally without forgetting | Data stream | Medium | Ongoing |
| [Full Fine-tune](#full-fine-tuning) | Train all model weights | Large dataset | Very High | Days-Weeks |

---

## RAG

**Retrieval-Augmented Generation**

Retrieve relevant documents/context before generating a response. No training required - works
immediately once you have embeddings.

### How It Works
```
Query → Embed → Vector Search → Top-K Documents → Inject into Prompt → Generate
```

### Data Required
- Content with embeddings (vector representations)
- No labeled data needed

### Compute
- Embedding generation: one-time, low cost
- Inference: just vector search + LLM call

### Pros
- Immediate results (no training)
- Easy to update (just add new documents)
- Transparent (can show retrieved sources)
- Works with any LLM (API or local)

### Cons
- Limited by retrieval quality
- Context window constraints
- Doesn't change model behavior, just context

### Industry Examples
- ChatGPT memory, Claude projects
- Perplexity, Bing Chat (search + generate)
- Enterprise knowledge bases

---

## Memory/Context Injection

**Structured Facts in System Prompt**

Store user preferences/facts and inject them into the system prompt. Simpler than RAG - no
vector search required.

### How It Works
```
User Facts DB → Format as Text → Prepend to System Prompt → Generate
```

### Data Required
- Structured facts (name, preferences, past decisions)
- Key-value or simple schema

### Compute
- None (just string formatting)

### Pros
- Zero latency to implement
- Deterministic (same facts = same context)
- Easy to debug and edit

### Cons
- Doesn't scale (limited by context window)
- Manual curation often needed
- No semantic understanding of relevance

### Industry Examples
- ChatGPT custom instructions
- Character.AI personality prompts

---

## Classifiers

**Small Models for Scoring/Filtering**

Train lightweight models (BERT, DistilBERT, or gradient boosting) to score or classify content.
Fast inference, cheap to train.

### How It Works
```
Labeled Data → Fine-tune Small Model → Predict Scores/Classes
```

### Data Required
- Labeled examples (100-10,000+)
- Binary labels (relevant/not) or multi-class

### Compute
- Training: Single GPU, hours
- Inference: CPU or GPU, milliseconds

### Pros
- Fast inference
- Cheap to train and host
- Interpretable (can extract feature importance)
- Works offline

### Cons
- Requires labeled data
- Fixed classes (can't handle new categories easily)
- Limited expressiveness compared to LLMs

### Industry Examples
- Spam filters
- Content moderation (toxicity classifiers)
- Recommendation pre-filtering

---

## Embedding Models

**Learning Similarity Representations**

Train or use pre-trained embedding models to represent content in vector space. Similar items
are close together.

### How It Works
```
Content → Embedding Model → 768-1536 dim vector → Store/Compare
```

### Data Required
- For pre-trained: nothing (just use off-the-shelf)
- For fine-tuned: positive pairs or triplets

### Compute
- Using pre-trained: Low
- Fine-tuning: Medium (single GPU, hours)

### Pros
- Enable semantic search
- Transfer across tasks (embedding is reusable)
- Can cluster without labels

### Cons
- Quality depends on model choice
- Fine-tuning requires paired data
- Not interpretable

### Industry Examples
- OpenAI embeddings, Cohere embeddings
- Sentence-BERT for semantic search
- E-commerce product similarity

---

## Prompt Tuning

**Learning Soft Prompt Tokens**

Instead of changing model weights, learn continuous prompt embeddings that steer the model.
Much cheaper than fine-tuning.

### How It Works
```
Learnable Tokens [P1][P2][P3] + User Prompt → Model → Output
```

### Data Required
- Examples of desired behavior (100-1000+)

### Compute
- Low (only learning small number of parameters)

### Pros
- Very parameter-efficient
- Easy to switch between "prompts"
- Works with frozen model

### Cons
- Less expressive than fine-tuning
- Requires access to model embeddings (not available via API)
- Can be unstable

### Industry Examples
- Research settings mainly
- Multi-task adapters in some production systems

---

## LoRA/Adapters

**Low-Rank Adaptation**

Add small trainable matrices to frozen model layers. Train these instead of full model.
Best balance of efficiency and capability for LLM customization.

### How It Works
```
Frozen Model Weights + Small LoRA Matrices (r=8-64) → Combined Output
LoRA updates: ~0.1% of parameters
```

### Data Required
- Instruction-response pairs (for SFT-style)
- Preference pairs (for DPO-style)
- 500-10,000+ examples typical

### Compute
- Training: 1-4 GPUs, 4-24 hours depending on model size
- Inference: Same as base model + tiny overhead

### Pros
- Efficient training (10-100x less compute than full fine-tune)
- Composable (can merge/swap adapters)
- Can be hot-swapped at inference
- Works with quantized models

### Cons
- Requires GPU access (not API models)
- Less capable than full fine-tune
- Hyperparameter sensitive (rank, alpha)

### Industry Examples
- Hugging Face PEFT library
- Most open-source model customization
- Private enterprise LLM deployments

---

## SFT (Supervised Fine-Tuning)

**Instruction Following from Examples**

Train model to follow instructions by showing input-output pairs. The first stage of most
LLM training pipelines (pre-training → SFT → RLHF).

### How It Works
```
(Instruction, Response) pairs → Cross-entropy Loss → Updated Model
```

### Data Required
- High-quality instruction-response pairs
- Typically 10,000-100,000+ examples

### Compute
- LoRA SFT: Medium (hours-days)
- Full SFT: High (days-weeks, multiple GPUs)

### Pros
- Teaches specific behaviors/formats
- Straightforward training objective
- Well-understood

### Cons
- Doesn't teach preferences (just mimics examples)
- Requires curated data
- Can overfit to training format

### Industry Examples
- Alpaca, Vicuna (instruction-tuned LLMs)
- ChatGPT stage 1
- Most fine-tuned models

---

## DPO (Direct Preference Optimization)

**Preference Learning Without Reward Model**

Train model to prefer "chosen" over "rejected" responses directly, without explicit reward
model. Simpler than RLHF, often similar quality.

### How It Works
```
(Prompt, Chosen, Rejected) triples → DPO Loss → Model prefers Chosen
```

### Data Required
- Preference pairs (500-10,000+)
- Each example: context + winning response + losing response

### Compute
- Medium (similar to SFT, can use LoRA)

### Pros
- Simpler than RLHF (no reward model)
- Stable training
- Direct optimization of preference objective

### Cons
- Requires explicit preference pairs
- Can't learn from scalar rewards
- Sensitive to reference model choice

### Industry Examples
- Zephyr, Neural Chat
- Many research preference-tuned models
- Increasingly replacing RLHF in practice

---

## DPO Variants (ORPO, KTO, IPO)

**Improved Preference Optimization Methods**

Newer algorithms that address DPO limitations - simpler objectives, better stability, or reduced
data requirements.

### ORPO (Odds Ratio Preference Optimization)

Combines SFT and preference optimization in one step.
```
Loss = SFT_loss + λ * odds_ratio_loss
```

**Pros:**
- Single training stage (no separate SFT)
- More stable than DPO
- Better sample efficiency

### KTO (Kahneman-Tversky Optimization)

Doesn't require paired preferences - works with binary good/bad labels.
```
Input: (prompt, response, is_good: bool)
```

**Pros:**
- No need for chosen/rejected pairs
- Works with thumbs up/down feedback
- Aligned with human loss aversion

**Cons:**
- Newer, less battle-tested
- May need more data than DPO

### IPO (Identity Preference Optimization)

Fixes theoretical issues with DPO's objective function.

**Pros:**
- More theoretically grounded
- Better generalization in some settings

### When to Use

| Method | Best For | Data Requirement |
|--------|----------|------------------|
| DPO | Standard preference tuning | Paired comparisons |
| ORPO | Combined SFT + alignment | Paired comparisons |
| KTO | Binary feedback only | Good/bad labels |
| IPO | Research, edge cases | Paired comparisons |

### Industry Examples
- ORPO: Used in recent open-source models (Mistral community)
- KTO: Gaining adoption for simpler feedback collection
- IPO: Primarily research settings

---

## RLHF (Reinforcement Learning from Human Feedback)

**Reward Model + Policy Optimization**

Train a reward model from comparisons, then optimize the LLM policy to maximize reward.
The original ChatGPT alignment approach.

### How It Works
```
1. Collect comparisons: "A is better than B"
2. Train reward model: R(prompt, response) → score
3. PPO/policy training: maximize E[R(response)]
```

### Data Required
- Comparison data (which response is better)
- Prompts for policy training

### Compute
- High (reward model + policy training + KL regularization)
- Multiple GPUs, days-weeks

### Pros
- Can learn nuanced preferences
- Works with scalar feedback
- Most flexible (arbitrary reward functions)

### Cons
- Complex pipeline (multiple models)
- Reward hacking possible
- Expensive and unstable
- Often overkill (DPO often sufficient)

### Industry Examples
- ChatGPT, Claude (original training)
- InstructGPT paper
- Mostly replaced by DPO/simpler methods now

---

## RLAIF (Reinforcement Learning from AI Feedback)

**AI-Generated Feedback for Alignment**

Replace human labelers with an AI model that provides feedback. The AI "judge" evaluates responses
and generates training signal.

### How It Works
```
1. Generate responses to prompts
2. AI judge scores/compares responses (using criteria)
3. Train reward model or use directly for DPO
4. Optimize policy against AI feedback
```

### Data Required
- Prompts (no human labels needed)
- Well-crafted judging criteria/rubrics
- Access to capable judge model

### Compute
- Medium-High (judge inference + training)
- Can be cheaper than human labeling at scale

### Pros
- Scales better than human feedback
- Consistent evaluation criteria
- Can iterate faster (no labeler latency)
- Cost-effective for large-scale training

### Cons
- AI judge has its own biases
- Can amplify model blind spots
- Quality ceiling limited by judge capability
- Constitutional AI concerns (self-referential)

### Industry Examples
- Anthropic Constitutional AI
- Google's RLAIF research
- Many open-source alignment projects
- LLM-as-judge evaluation frameworks

---

## Reward Modeling

**Training Scalar Reward Predictors**

Train a model to predict human preferences as a scalar score. Used standalone or as part of RLHF.

### How It Works
```
(Prompt, Response_A, Response_B, Winner) → Reward Model → R(prompt, response) → scalar
```

### Data Required
- Pairwise comparisons (which response is better)
- Typically 10,000-100,000+ comparisons

### Compute
- Medium (similar to classifier training)
- Often uses same architecture as policy model

### Pros
- Provides dense reward signal
- Can score any response (not just training data)
- Enables reinforcement learning optimization
- Interpretable (can inspect what gets high/low scores)

### Cons
- Reward hacking (model games the reward)
- Distribution shift (reward model trained on different responses)
- Requires careful calibration
- Can be expensive to maintain

### Use Cases
- RLHF training (the "R" in RLHF)
- Response ranking at inference
- Quality filtering for training data
- A/B testing automation

### Industry Examples
- OpenAI reward models (InstructGPT)
- Anthropic preference models
- Open-source: trl RewardTrainer

---

## Knowledge Distillation

**Transfer Knowledge to Smaller Models**

Train a smaller "student" model to mimic a larger "teacher" model. Get most of the capability at
fraction of the cost.

### How It Works
```
Teacher (large) → Soft Labels/Outputs → Student (small) learns to match
```

### Types
1. **Output distillation:** Student matches teacher's output distribution
2. **Feature distillation:** Student matches intermediate representations
3. **Synthetic data:** Teacher generates training data for student

### Data Required
- Prompts/inputs to run through teacher
- Teacher model outputs (logits or text)
- No human labels needed

### Compute
- Medium (teacher inference + student training)
- Student training is cheap; teacher inference can be expensive

### Pros
- Dramatically smaller deployment cost
- Faster inference (smaller model)
- Can distill API models into local models
- Captures implicit knowledge

### Cons
- Student ceiling limited by teacher
- May lose nuanced capabilities
- Some capabilities don't transfer well
- Legal considerations with API terms

### Industry Examples
- Alpaca (distilled from GPT-3.5)
- Orca, Phi models (synthetic data from GPT-4)
- DistilBERT, TinyBERT
- Most "small but capable" open models

---

## Synthetic Data Generation

**Using LLMs to Create Training Data**

Use capable LLMs to generate training examples, augment existing data, or create entirely new
datasets.

### How It Works
```
Seed Examples + Prompt Engineering → LLM → Synthetic Training Data → Train Model
```

### Types
1. **Self-instruct:** Model generates its own instruction-following data
2. **Evol-instruct:** Iteratively make instructions more complex
3. **Data augmentation:** Paraphrase, expand, or diversify existing data
4. **Domain synthesis:** Generate domain-specific examples from rules

### Data Required
- Seed examples (few to hundreds)
- Clear generation templates/prompts
- Quality filtering criteria

### Compute
- Low-Medium (LLM inference for generation)
- Quality filtering adds overhead

### Pros
- Scales beyond human labeling capacity
- Cheap compared to manual annotation
- Can target specific gaps/domains
- Fast iteration on data composition

### Cons
- Quality ceiling limited by generator
- Can amplify biases
- Synthetic data collapse (training on own outputs)
- May lack diversity without careful design

### Industry Examples
- Self-Instruct, Alpaca
- WizardLM (Evol-instruct)
- Phi models (textbook-quality synthetic data)
- Most modern instruction-tuned models use some synthetic data

---

## Active Learning

**Selecting Most Informative Samples**

Instead of labeling all data, intelligently select which examples to label for maximum learning
efficiency.

### How It Works
```
1. Train initial model on small labeled set
2. Score unlabeled data by uncertainty/informativeness
3. Select most valuable samples for labeling
4. Label and retrain
5. Repeat
```

### Selection Strategies
- **Uncertainty sampling:** Label examples model is least confident about
- **Diversity sampling:** Label examples that are most different from current data
- **Expected model change:** Label examples that would most change the model
- **Committee disagreement:** Label examples where ensemble disagrees

### Data Required
- Small initial labeled set
- Large unlabeled pool
- Human labelers (or AI judge) in the loop

### Compute
- Low (just model inference for scoring)
- Main cost is labeling, not compute

### Pros
- Dramatically reduces labeling cost
- Focuses effort on hard/informative examples
- Better data efficiency
- Works with any model type

### Cons
- Requires labeling infrastructure
- Can miss important edge cases
- Selection bias possible
- Overhead of selection process

### Industry Examples
- Production ML systems with ongoing labeling
- Annotation platforms (Scale AI, Labelbox)
- Medical imaging, autonomous vehicles
- Any system with expensive labeling

---

## Continual Learning

**Learning Incrementally Without Forgetting**

Update models on new data without catastrophic forgetting of previous knowledge.

### How It Works
```
Model_v1 → New Data → Update → Model_v2 (retains v1 knowledge)
```

### Approaches
1. **Replay:** Mix new data with samples from old data
2. **Regularization:** Penalize changes to important weights (EWC, SI)
3. **Architecture:** Dedicated parameters for new tasks
4. **Knowledge distillation:** Use old model as teacher

### Data Required
- Stream of new data
- Access to (samples of) old data for replay
- Task boundaries (sometimes)

### Compute
- Medium (similar to regular training)
- Replay adds storage/retrieval overhead

### Pros
- Model improves over time
- Don't need to retrain from scratch
- Adapts to distribution shift
- Enables targeted customization over time

### Cons
- Catastrophic forgetting is hard to prevent
- Need to balance old vs new
- Evaluation is complex
- May accumulate errors

### Industry Examples
- Recommendation systems (user preferences change)
- Spam filters (adversarial evolution)
- Personal assistants (learning user patterns)
- Any system that needs ongoing updates

---

## Full Fine-Tuning

**Training All Model Weights**

Update every parameter in the model. Maximum capability but maximum cost.

### How It Works
```
Dataset → Forward/Backward on all weights → Fully adapted model
```

### Data Required
- Large, high-quality dataset (100,000+ examples typical)

### Compute
- Very High (8+ GPUs, days-weeks for 7B+ models)
- 4-8x memory of inference

### Pros
- Maximum adaptation capability
- Can fundamentally change model behavior
- Best for very specialized domains

### Cons
- Extremely expensive
- Catastrophic forgetting risk
- Hard to iterate quickly
- May not be necessary (LoRA often sufficient)

### Industry Examples
- Domain-specific models (medical, legal)
- Language adaptation
- Creating new base models

---

## Comparison Matrix

| Approach | Needs Training | Needs GPU | API Compatible | Data Volume | Time to Deploy |
|----------|----------------|-----------|----------------|-------------|----------------|
| RAG | No | No | Yes | Any | Hours |
| Memory | No | No | Yes | Small | Minutes |
| Classifiers | Yes | Optional | N/A | 100+ | Hours |
| Embeddings | Optional | Optional | Yes | Any | Hours |
| Prompt Tuning | Yes | Yes | No | 100+ | Hours |
| LoRA | Yes | Yes | No | 500+ | Hours-Days |
| SFT | Yes | Yes | No | 1000+ | Days |
| DPO | Yes | Yes | No | 500+ | Days |
| DPO Variants | Yes | Yes | No | 500+ | Days |
| RLHF | Yes | Yes | No | 1000+ | Weeks |
| RLAIF | Yes | Yes | No | 1000+ | Days-Weeks |
| Reward Modeling | Yes | Yes | No | 10000+ | Hours-Days |
| Distillation | Yes | Optional | Yes* | 1000+ | Days |
| Synthetic Data | No | No | Yes | 10+ seeds | Hours |
| Active Learning | Yes | Optional | N/A | Varies | Ongoing |
| Continual Learning | Yes | Yes | No | Stream | Ongoing |
| Full Fine-tune | Yes | Yes | No | 10000+ | Weeks |

*Distillation can use API models as teacher

---

## Decision Framework

**Start simple, add complexity only when needed:**

1. **Immediate customization needed?** → RAG or Memory
2. **Need to score/filter content?** → Classifiers
3. **Need similarity search?** → Embeddings
4. **Need training data?** → Synthetic Data Generation
5. **Limited labeling budget?** → Active Learning
6. **Need to change LLM behavior?** → LoRA + DPO (or KTO for binary feedback)
7. **Want smaller deployment model?** → Knowledge Distillation
8. **System needs ongoing updates?** → Continual Learning
9. **Have massive data + compute?** → Full fine-tune
10. **Need complex alignment?** → RLHF or RLAIF (rarely needed)

Most production systems combine multiple approaches:
- RAG for context
- Classifiers for filtering
- LoRA for style/preference
- Synthetic data to bootstrap training

---

## Out of Scope

The following approaches are **not covered** in this framework because they require resources or
infrastructure beyond what's practical for customization use cases. We document them here for
completeness and to explain why we don't use them.

---

### Training from Scratch

**Pre-training a new base model from random initialization**

Create a completely new language model by training on massive text corpora. This is how models like
GPT-4, Claude, and Llama are initially created.

#### How It Works
```
Random Weights → Trillions of Tokens → Next-Token Prediction → Base Model
```

The model learns language, facts, reasoning patterns, and capabilities from scratch through
next-token prediction on internet-scale data.

#### What It's For
- Creating new foundation models with specific capabilities
- Building models with different architectures (context length, efficiency)
- Training on proprietary data that can't be shared with existing models
- Research into model capabilities and emergent behaviors

#### Requirements

| Aspect | Requirement |
|--------|-------------|
| Data | 1-15+ trillion tokens (curated web, books, code) |
| Compute | 1,000-10,000+ GPUs for weeks/months |
| Cost | $1M-$100M+ |
| Expertise | Large ML team (10-100+ researchers) |
| Infrastructure | Custom training frameworks, distributed systems |

#### Industry Examples
- GPT-4 (OpenAI)
- Claude (Anthropic)
- Llama 2/3 (Meta)
- Gemini (Google)
- Mistral, Qwen, Yi (various)

#### Why Out of Scope
We use existing base models and adapt them. Pre-training is for organizations with massive compute
budgets and research teams. The open-source ecosystem provides excellent base models we can build
on.

---

### Continued Pre-training (Full)

**Extended pre-training on domain-specific corpus**

Take an existing base model and continue pre-training on domain-specific text to inject specialized
knowledge before fine-tuning.

#### How It Works
```
Base Model → Domain Corpus (billions of tokens) → Continued Next-Token Prediction → Domain Model
```

The model's weights are updated to better represent domain-specific language, concepts, and
patterns while (hopefully) retaining general capabilities.

#### What It's For
- Domain adaptation: medical, legal, financial, scientific
- Language adaptation: training English model on other languages
- Code specialization: training on specific programming languages/frameworks
- Temporal updates: incorporating recent knowledge

#### Requirements

| Aspect | Requirement |
|--------|-------------|
| Data | 10B-100B+ domain tokens |
| Compute | 64-256+ GPUs for days |
| Cost | $10K-$1M |
| Risk | Catastrophic forgetting of general capabilities |

#### Industry Examples
- BioMedLM (Stanford - medical)
- CodeLlama (Meta - code)
- Galactica (Meta - scientific, discontinued)
- BloombergGPT (financial)
- Various legal/medical LLMs

#### Why Out of Scope
For customization, LoRA + domain-specific RAG achieves 80% of the benefit at 1% of the cost.
Continued pre-training makes sense when building a domain foundation model for broad use, not for
targeted customization.

**Alternative we use:** Domain-specific RAG with curated knowledge bases.

---

### Architecture Research

**Designing novel model architectures and training objectives**

Research into new attention mechanisms, positional encodings, model structures, or training
approaches that could improve model capabilities or efficiency.

#### What It Includes
- New attention patterns (sparse, linear, sliding window)
- Alternative architectures (state space models like Mamba)
- Novel positional encodings (RoPE, ALiBi, NTK-aware)
- Training objective innovations (beyond next-token prediction)
- Efficiency improvements (quantization-aware training, pruning)

#### What It's For
- Improving context length (handling longer documents)
- Reducing inference cost (faster, cheaper generation)
- Better reasoning capabilities
- More efficient training
- Novel capabilities (better math, code, multilingual)

#### Requirements

| Aspect | Requirement |
|--------|-------------|
| Expertise | PhD-level ML researchers |
| Compute | Extensive experimentation budget |
| Timeline | Months to years of research |
| Validation | Extensive benchmarking and ablations |

#### Industry Examples
- Transformers (Google, 2017)
- Flash Attention (Dao et al.)
- Mixture of Experts (Google, OpenAI)
- State Space Models (Mamba)
- RoPE, ALiBi (positional encoding improvements)

#### Why Out of Scope
We use proven architectures. Architecture research is for ML researchers at labs and universities.
The innovations eventually appear in open-source models we can adopt.

**Our approach:** Use best available open-source models with proven architectures.

---

### Multi-Modal Training

**Training models that understand multiple modalities (text + images, audio, video)**

Create or adapt models to process and generate across different input/output types.

#### How It Works
```
Image Encoder + Text Model → Joint Training → Vision-Language Model
Audio Encoder + Text Model → Joint Training → Speech-Language Model
```

Requires aligning representations across modalities so the model understands relationships between
(e.g.) images and their descriptions.

#### What It's For
- Image understanding and generation (describe images, answer questions)
- Document understanding (PDFs, charts, diagrams)
- Video analysis (summarization, search)
- Speech recognition and synthesis
- Embodied AI (robotics, autonomous systems)

#### Requirements

| Aspect | Requirement |
|--------|-------------|
| Data | Millions of aligned pairs (image-caption, audio-transcript) |
| Compute | High (multiple encoders + language model) |
| Expertise | Specialized in each modality |
| Infrastructure | Multi-modal data pipelines |

#### Industry Examples
- GPT-4V, GPT-4o (OpenAI)
- Claude 3 vision (Anthropic)
- Gemini (Google - native multi-modal)
- LLaVA, Qwen-VL (open source)
- Whisper (OpenAI - speech)

#### Why Out of Scope
We focus on text-based customization. Multi-modal is a different problem space with different
infrastructure. If needed, we can use existing multi-modal APIs (GPT-4V, Claude vision) without
training our own.

**Our approach:** Use multi-modal APIs when needed; don't train multi-modal models.

---

### Mixture of Experts (MoE) Training

**Training sparse models where only a subset of parameters activate per input**

MoE models have multiple "expert" sub-networks and a router that selects which experts process each
token. This allows massive parameter counts with manageable compute.

#### How It Works
```
Input → Router → Select K of N Experts → Expert Processing → Combine Outputs
```

For example, Mixtral 8x7B has 8 expert networks but only uses 2 per token, giving 47B total
parameters but ~13B active parameters per forward pass.

#### What It's For
- Scaling model capacity without proportional compute increase
- Specialization (different experts for different tasks/domains)
- Better performance per FLOP
- Enabling very large models on limited hardware

#### Requirements

| Aspect | Requirement |
|--------|-------------|
| Architecture | Custom MoE layers, routing mechanisms |
| Training | Load balancing, expert utilization optimization |
| Inference | Efficient expert selection, memory management |
| Expertise | Deep understanding of sparse architectures |

#### Industry Examples
- Mixtral 8x7B, 8x22B (Mistral)
- GPT-4 (rumored MoE)
- Switch Transformer (Google)
- DeepSeek MoE
- Grok (xAI)

#### Why Out of Scope
MoE is a base model architecture choice. We can *use* MoE models (Mixtral is excellent) but don't
need to *train* them. The complexity of MoE training (load balancing, routing optimization) is
handled by base model creators.

**Our approach:** Use pre-trained MoE models; apply LoRA for customization just like dense models.

---

## Summary

This document covers the landscape of practical approaches for LLM customization. The key insight is
that **most value comes from simple approaches** (RAG, classifiers, embeddings) with training-based
methods (LoRA, DPO) reserved for when you need the model itself to change behavior.
