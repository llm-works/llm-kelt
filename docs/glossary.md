# Glossary

Terms with a specific meaning in llm-kelt. Generic ML terms (LoRA, DPO, SFT, HNSW, RLHF, ...)
are assumed background — see the HuggingFace, PEFT, or TRL docs for those.

| Term | Meaning |
|---|---|
| **assertion** | A free-text `Fact` of type `"assertion"`. The type `ContextBuilder` injects into prompts. |
| **atomic memory** | The `Fact`-row + details-table subsystem under `kelt.atomic.*`. Seven typed clients (assertions, feedback, preferences, predictions, directives, interactions, solutions) sharing one row per record. |
| **context_key** | Flat isolation key on every atomic write. Reads filter exactly (or with SQL LIKE globs from a read-only client). Not hierarchical. |
| **scope_key** | Hierarchical key on every KG record. Reads walk up the chain (`"org:acme:user:alice"` sees `"org:acme"` and `"global"`). Writes always land in the exact key. |
| **schema_name** | Postgres schema hosting the tables. Different schemas = physical separation. Set via `ClientContext.schema_name` or `client.with_schema()`. |
| **SchemaMode** | What the client does on startup: `ENSURE` runs migrations, `VERIFY` checks version, `SKIP` doesn't touch alembic (and skips the pgvector import). |
| **manifest** | YAML file describing a training run: adapter, method, data source, LoRA/training config, deployment policy. Persisted to `<registry>/pending/` then `<registry>/completed/`. |
| **adapter (registry)** | A named series of trained versions. Each version is identified by a 12-char md5. Deploys are per-version. |
| **fact ID** | `int` returned from `add()` / `record()` methods. Primary key on `memv1_facts`. |
| **signal** | Feedback direction: `"positive"`, `"negative"`, `"dismiss"`. |
| **strength** | Feedback intensity, `0.0`–`1.0`. Used as a filter on training exports (`min_strength=`). |
| **margin** | Preference-pair confidence. Used as a filter on DPO exports (`min_margin=`). |
| **quantized embedding** | Fact vector stored as `F32`, `F16` (halfvec, default), `I8`, or `I4`. Format picks the table (`embeddings_f16_384`, etc.). |
| **details table** | Per-fact-type table joined to `memv1_facts` (`memv1_feedback_details`, `memv1_preference_details`, ...). Eagerly loaded when clients return `Fact` objects. |
