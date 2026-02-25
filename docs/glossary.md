# Glossary

Quick definitions of terms used in llm-kelt.

---

| Term | Definition |
|------|------------|
| **Adapter** | Small trainable module added to frozen model (see LoRA) |
| **Base Model** | Pre-trained model before fine-tuning |
| **Calibration** | Measuring if confidence matches actual accuracy |
| **Checkpoint** | Saved model weights at a point in training |
| **Cosine Distance** | Measure of angle between vectors (0=identical, 1=opposite) |
| **DPO** | Direct Preference Optimization - train on chosen/rejected pairs |
| **Embedding** | Dense vector representation of text/content |
| **Fine-tuning** | Adapting pre-trained model to specific task |
| **HNSW** | Hierarchical Navigable Small World - vector index type |
| **IVFFlat** | Inverted File Flat - vector index type for pgvector |
| **JSONL** | JSON Lines - one JSON object per line |
| **KL Divergence** | Measure of difference between probability distributions |
| **LoRA** | Low-Rank Adaptation - efficient fine-tuning method |
| **Margin** | How much one option is preferred over another |
| **pgvector** | PostgreSQL extension for vector similarity search |
| **PPO** | Proximal Policy Optimization - RL algorithm |
| **Pre-training** | Initial training on large corpus (internet) |
| **RAG** | Retrieval-Augmented Generation - fetch docs before generating |
| **Rank** | Dimension of LoRA matrices (r=8 means 8 dimensions) |
| **Reward Model** | Model that predicts human preferences (RLHF) |
| **RLHF** | Reinforcement Learning from Human Feedback |
| **Semantic Search** | Search by meaning, not keywords |
| **SFT** | Supervised Fine-Tuning - first stage of RLHF |
| **Signal** | Feedback type: positive, negative, dismiss |
| **Strength** | Intensity of feedback signal (0.0-1.0) |
| **Transfer Learning** | Reusing knowledge from one task for another |
| **Vector** | Array of floats representing content in embedding space |
| **Vector Index** | Data structure for fast similarity search |
