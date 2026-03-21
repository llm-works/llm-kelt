# CLI Reference

The `kelt` CLI provides commands for training management and adapter operations.

## Training Commands

All training commands are under `kelt train` (alias: `kelt t`).

### Run Training

Execute a training manifest:

```bash
# Interactive selection from pending manifests
kelt train run

# Run specific manifest
kelt train run manifests/my-adapter.yaml

# Override model
kelt train run manifests/my-adapter.yaml --model Qwen2.5-7B-Instruct

# Skip adapter registration after training
kelt train run manifests/my-adapter.yaml --skip-register

# Manual LoRA profile (auto-detected by default)
kelt train run manifests/my-adapter.yaml --lora-profile large

# List available models
kelt train run --list-models
```

**Options:**
| Flag | Description |
|------|-------------|
| `--model, -m` | Override model (path, HF ID, or configured name) |
| `--list-models` | List available models and exit |
| `--skip-register` | Skip adapter registration after training |
| `--lora-profile` | Manual profile override: `small`, `medium`, `large`, `xlarge` |

---

### List Manifests

List pending or completed training manifests:

```bash
# List pending manifests
kelt train list

# List completed manifests
kelt train list --completed
```

**Aliases:** `kelt train l`, `kelt train ls`

---

### Show Manifest

Display manifest details:

```bash
# Show by path
kelt train show manifests/my-adapter.yaml

# Show by name (looks in pending/)
kelt train show my-adapter
```

---

### Direct SFT Training

Train an SFT adapter directly from JSONL data (bypasses manifest workflow):

```bash
kelt train sft \
  --data train.jsonl \
  --output ./my-adapter \
  --model Qwen2.5-7B-Instruct \
  --epochs 3 \
  --lr 2e-4

# Resume from checkpoint
kelt train sft --data train.jsonl --output ./my-adapter --based-on ./checkpoint

# Disable quantization (full precision)
kelt train sft --data train.jsonl --output ./my-adapter --no-quantize
```

**Aliases:** `kelt train s`

**Options:**
| Flag | Description |
|------|-------------|
| `--data, -d` | Input JSONL path (required) |
| `--output, -o` | Output adapter directory (required) |
| `--model, -m` | Model name or path |
| `--epochs` | Training epochs |
| `--lr` | Learning rate |
| `--based-on, -b` | Resume from checkpoint path |
| `--no-quantize` | Disable QLoRA quantization |

**Data format** (JSONL):
```json
{"instruction": "What is 2+2?", "output": "The answer is 4."}
{"instruction": "Say hello.", "output": "Hello!"}
```

---

### Direct DPO Training

Train a DPO adapter directly from JSONL data:

```bash
kelt train dpo \
  --data preferences.jsonl \
  --output ./my-dpo-adapter \
  --model Qwen2.5-7B-Instruct \
  --beta 0.1

# Stack on existing adapter
kelt train dpo --data prefs.jsonl --output ./stacked --based-on ./parent-adapter
```

**Aliases:** `kelt train d`

**Options:**
| Flag | Description |
|------|-------------|
| `--data, -d` | Input JSONL path (required) |
| `--output, -o` | Output adapter directory (required) |
| `--model, -m` | Model name or path |
| `--beta` | DPO beta parameter (default: 0.1) |
| `--epochs` | Training epochs |
| `--lr` | Learning rate |
| `--based-on, -b` | Parent adapter for stacking |
| `--no-quantize` | Disable QLoRA quantization |

**Data format** (JSONL):
```json
{"prompt": "Explain X", "chosen": "Clear explanation...", "rejected": "Confusing explanation..."}
```

---

### List Adapters

List registered adapters:

```bash
# List all adapters
kelt train adapters

# Show only deployed adapters
kelt train adapters --deployed
```

**Aliases:** `kelt train a`

---

### Deploy Adapter

Deploy an adapter version to make it available for inference:

```bash
# Deploy latest version
kelt train deploy my-adapter

# Deploy specific version (by version ID or md5 prefix)
kelt train deploy my-adapter --version 20260315-120000-abc123

# Deploy with md5 prefix matching
kelt train deploy my-adapter --version abc123

# Add deployment (keep existing)
kelt train deploy my-adapter --policy add

# Replace existing deployment (default)
kelt train deploy my-adapter --policy replace

# Clear all deployments for adapter
kelt train deploy my-adapter --clear
```

**Aliases:** `kelt train dp`

**Options:**
| Flag | Description |
|------|-------------|
| `--version, -v` | Version ID or md5 prefix (latest if omitted) |
| `--policy, -p` | `add` or `replace` (default: replace) |
| `--clear, -c` | Remove all deployments for adapter |

---

### Merge Adapter

Merge a LoRA adapter into base model weights (creates a new model):

```bash
# Merge by adapter path
kelt train merge ./adapters/my-adapter

# Merge deployed adapter by name
kelt train merge my-adapter

# Merge by md5 hash
kelt train merge abc123def456

# Specify output path
kelt train merge my-adapter --output ./merged-model

# Override base model
kelt train merge my-adapter --model Qwen2.5-7B-Instruct

# Specify output dtype
kelt train merge my-adapter --dtype float16
```

**Aliases:** `kelt train m`

**Options:**
| Flag | Description |
|------|-------------|
| `--model, -m` | Base model name (auto-detected from adapter if omitted) |
| `--output, -o` | Output path (default: `<model>-<md5>`) |
| `--dtype` | Output dtype: `bfloat16`, `float16`, `float32` (default: bfloat16) |
| `--overwrite` | Overwrite existing output without prompting |

**Use cases:**
- VLM models where vLLM doesn't apply LoRA correctly
- Creating standalone fine-tuned models for deployment
- Reducing inference overhead by baking in the adapter

---

## Proxy Server

Start the context-injecting proxy server:

```bash
kelt proxy
```

The proxy intercepts LLM requests and injects relevant context from the kelt database.

---

## Configuration

The CLI reads configuration from:

1. `etc/llm-kelt.yaml` (main config)
2. `.env.yaml` (local overrides, gitignored)

Key configuration sections:

```yaml
# Model locations for training
models:
  locations:
    - ~/ops/models/hf
  selection:
    generate:
      default: Qwen2.5-7B-Instruct

# Adapter registry path
adapters:
  lora:
    base_path: ~/ops/models/adapters/peft

# Training defaults by method
training:
  default_profiles:
    sft:
      epochs: 3
      batch_size: 4
      learning_rate: 0.0002
    dpo:
      beta: 0.1
      epochs: 3
```
