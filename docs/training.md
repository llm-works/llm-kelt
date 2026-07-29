# Training

Fine-tuning workflow: export recorded feedback and preferences to training data, train an
adapter, register it, deploy it to an inference server.

Two paths:

- **Direct** — call `train_lora(...)` or `train_dpo(...)` from Python. Fast iteration.
- **Manifest** — write a YAML file describing the training run, submit it, run it with
  `kelt train run <manifest>`. Reproducible, file-based, works well from CI.

Both produce the same `RunResult` and register the same way. Pick manifest for anything
you'll re-run.

Install the training extras first:

```bash
pip install llm-kelt[training]
```

That pulls in `torch`, `transformers`, `peft`, `trl`, `bitsandbytes`, and datasets.

## Config sections

Training reads two extra config sections:

```yaml
# etc/llm-kelt.yaml
kelt:
  adapters:
    lora:
      base_path: ~/models/adapters         # required
      infer_url: http://localhost:8000     # optional — for registry refresh
  training:
    default_profile: medium                # optional; auto-detected from model size otherwise

models:
  locations:                               # for CLI resolution of base_model
    - ~/models/huggingface
```

`kelt.adapters.lora.base_path` is required — the property `kelt.train` raises
`RuntimeError` without it.

## Exporting training data

Recorded feedback and preferences → JSONL files ready for HF datasets.

```python
from llm_kelt.training import (
    ExportResult,
    export_feedback_sft,
    export_feedback_classifier,
)
from llm_kelt.training.dpo import export_preferences
```

### SFT export

```python
def export_feedback_sft(
    session_factory,
    context_key: str | None,
    output_path: Path | str,
    *,
    signal: Literal["positive", "negative", "dismiss"] = "positive",
    min_strength: float = 0.5,
    since: datetime | None = None,
    until: datetime | None = None,
    include_context: bool = False,
) -> ExportResult
```

Emits one JSONL row per matching feedback record:

```
{"instruction": "…prompt…", "output": "…content…"}
```

With `include_context=True`, an `"input"` field is added — Alpaca format for context-aware
prompts. Example:

```python
result = export_feedback_sft(
    session_factory=kelt.database.session,
    context_key=kelt.context_key,
    output_path="feedback_sft.jsonl",
    signal="positive",
    min_strength=0.6,
)
print(result.path, result.count, result.format)
```

### Classifier export

```python
def export_feedback_classifier(
    session_factory, context_key, output_path,
    *, since=None, until=None, min_strength: float = 0.5,
) -> ExportResult
```

Emits `{"text": …, "label": 0|1}` — positive feedback → 1, negative → 0. For training a
binary quality classifier.

### DPO export

```python
from llm_kelt.training.dpo import export_preferences

export_preferences(
    session_factory=kelt.database.session,
    context_key=kelt.context_key,
    output_path="preferences_dpo.jsonl",
    category=None,  # filter to a single category
    since=None,
    until=None,  # time bounds
    min_margin=None,  # filter on the margin field
)
```

Emits `{"prompt": …, "chosen": …, "rejected": …}` per preference. Fed straight to TRL's
`DPOTrainer` or `train_dpo(...)`.

### `ExportResult`

```python
@dataclass
class ExportResult:
    path: Path
    count: int
    context_key: str | None
    format: str
    exported_at: datetime
```

## Direct training — LoRA / SFT

```python
from llm_kelt.training import train_lora
from llm_kelt.training.lora import Config as LoraConfig

lora_cfg = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    use_rslora=True,  # rank-stabilized: alpha/sqrt(r) instead of alpha/r
    bias="none",
    task_type="CAUSAL_LM",
)

result = train_lora(
    lg=lg,
    data_path="feedback_sft.jsonl",
    output_dir="./adapters/coding-v1",
    base_model="Qwen/Qwen2.5-7B-Instruct",
    lora_config=lora_cfg,
    training_config={
        "num_epochs": 3,
        "batch_size": 4,
        "gradient_accumulation_steps": 4,
        "learning_rate": 2e-4,
        "warmup_ratio": 0.03,
        "max_seq_length": 2048,
        "max_grad_norm": 1.0,
        "logging_steps": 10,
        "save_steps": 100,
        "fp16": True,
        "gradient_checkpointing": True,
        "neftune_noise_alpha": 5.0,  # optional: embedding noise regularization
    },
    quantize=True,  # 4-bit QLoRA — lower VRAM
)

print(result.adapter.path)
print(result.metrics["train_loss"])
```

`train_lora` returns a `RunResult`. Adapter path is `result.adapter.path`. Training metrics
come from HuggingFace's `Trainer` state (loss curve, learning rate, gradient norms).

## Direct training — DPO

```python
from llm_kelt.training import train_dpo

result = train_dpo(
    lg=lg,
    data_path="preferences_dpo.jsonl",
    output_dir="./adapters/aligned-v1",
    base_model="Qwen/Qwen2.5-7B-Instruct",
    lora_config=LoraConfig(r=16, lora_alpha=32),
    training_config={"num_epochs": 3, "batch_size": 2, "learning_rate": 5e-5},
    method_config={"beta": 0.1},  # DPO temperature — lower = closer to reference model
    quantize=True,
)
```

`method_config["beta"]` is the DPO KL-regularisation strength.

## Manifest workflow

For anything reproducible. Manifest describes: which adapter series, method, data source,
config overrides, deployment policy.

### Manifest schema

```python
@dataclass
class Manifest:
    adapter: str  # series name, e.g. "coding-v1"
    method: Literal["dpo", "sft", "prompt"]
    data: Data  # inline records or external path
    deployment: Deployment | None = None
    version: str = "1"
    created_at: datetime = now()  # auto-populated
    source: Source | None = None
    parent: Adapter | None = None  # continue-training from a prior adapter
    lora: DotDict = {}  # profile-detected defaults
    training: DotDict = {}  # profile-detected defaults
    method_config: DotDict = {}  # {"beta": …} for DPO, {"num_virtual_tokens": …} for prompt
    output: RunResult | None = None  # populated after run
```

### Create programmatically

```python
manifest = kelt.train.manifest.create(
    adapter="coding-v1",
    method="sft",
    data=[
        {"instruction": "What is 2+2?", "output": "4."},
        {"instruction": "Reverse 'abc'.", "output": "'cba'"},
    ],
    model="Qwen/Qwen2.5-7B-Instruct",
    context_key=kelt.context_key,
    schema_name="production",
    description="First SFT pass on positive feedback",
    config={
        "lora": {"r": 16, "lora_alpha": 32},
        "training": {"num_epochs": 3, "learning_rate": 2e-4},
    },
    deployment_policy="add",  # "skip" | "add" | "replace"
)

kelt.train.manifest.save(manifest, Path("manifests/coding-v1.yaml"))
kelt.train.manifest.submit(manifest)  # moves to pending queue
```

`submit()` returns a `SubmitResult(adapter, timestamp, location)` — `location` is the path
the runner will read.

### Author by hand

```yaml
# manifests/coding-v1.yaml
adapter: coding-v1
method: sft
version: "1"

source:
  context_key: agent:code-reviewer
  schema_name: production
  description: First SFT pass on positive feedback

data:
  format: inline
  records:
    - instruction: What is 2+2?
      output: "4."
    - instruction: Reverse 'abc'.
      output: "'cba'"

# or external:
# data:
#   format: external
#   path: ./data/feedback_sft.jsonl

lora:
  r: 16
  lora_alpha: 32
  lora_dropout: 0.05

training:
  num_epochs: 3
  batch_size: 4
  learning_rate: 2.0e-4
  max_seq_length: 2048

deployment:
  policy: add    # "skip" | "add" | "replace"
```

### Run

```bash
kelt train run manifests/coding-v1.yaml
# or select from a picker:
kelt train run
# or programmatically:
```

```python
from llm_kelt.training import Runner
from llm_kelt.training.storage import FileStorage

storage = FileStorage(lg, base_path="~/models/adapters")
runner = Runner(lg, storage, model_locations=[Path("~/models/huggingface")])
result = runner.run(Path("manifests/coding-v1.yaml"))
```

The runner:

1. Loads and validates the manifest.
2. Resolves `base_model` via `ModelResolver` over `model_locations`.
3. Dispatches to `dpo.Client`, `sft.Client`, or `prompt.Client` based on `method`.
4. Enriches the `RunResult` with the adapter's `md5` (12-char hex — the version ID).
5. Optionally registers via `AdapterRegistry` and deploys per `deployment.policy`.
6. Writes the completed manifest to `~/models/adapters/<adapter>/manifests/completed/`.

Override the resolved model or profile for one run:

```bash
kelt train run manifests/coding-v1.yaml --model Qwen/Qwen2.5-14B-Instruct --lora-profile medium
```

`--skip-register` runs training but skips the registry write, useful for dry runs.

### Query the manifest store

```python
kelt.train.manifest.list_pending()  # → list[Manifest]
kelt.train.manifest.list_completed()  # → list[Manifest]
kelt.train.manifest.get_pending("coding-v1")  # → Manifest | None
kelt.train.manifest.remove_pending("coding-v1")
kelt.train.manifest.get_manifest("a1b2c3d4e5f6")  # by md5
kelt.train.manifest.find_adapter("a1b2c3d4e5f6")  # → Adapter | None
kelt.train.manifest.get_latest_completed(adapter="coding-v1")
```

### Continue-training from a parent adapter

```python
parent = kelt.train.manifest.find_adapter("a1b2c3d4e5f6")

manifest = kelt.train.manifest.create(
    adapter="coding-v2",
    method="dpo",
    data=[...],
    parent=parent,  # DPO chains cleanly from an SFT parent
    config={"method_config": {"beta": 0.1}},
)
```

The runner loads the parent adapter as the reference model and initialises the new LoRA
weights on top. Note: SFT explicitly does *not* chain from a `parent` yet — you'll get a
warning and training will proceed from the base model.

## Method-specific clients

Skip the runner and drive a method directly:

```python
result = kelt.train.dpo.train(manifest, register=True)
result = kelt.train.sft.train(manifest, register=True)
result = kelt.train.prompt.train(manifest, register=True)
```

Each validates that `manifest.method` matches, then runs the same LoRA/DPO/prompt-tuning
pipeline the runner would. `register=False` skips writing to the adapter registry.

## Prompt tuning

For very large models where LoRA is unstable. Trains a small number of soft prompt vectors
prepended to the input, base weights frozen.

Manifest:

```yaml
adapter: assistant-prompt-v1
method: prompt
data:
  format: external
  path: ./data/feedback_sft.jsonl
method_config:
  num_virtual_tokens: 8
  prompt_tuning_init: TEXT
  prompt_tuning_init_text: You are a helpful assistant.
training:
  num_epochs: 5
  learning_rate: 1.0e-3
```

Only `method_config` differs from an SFT manifest. Init from text vs random with
`prompt_tuning_init`.

## LoRA config

```python
class Config:  # llm_kelt.training.lora.Config
    r: int
    lora_alpha: int
    lora_dropout: float = 0.05
    target_modules: list[str] | Literal["all-linear"] = "all-linear"
    bias: Literal["none", "all", "lora_only"] = "none"
    task_type: str = "CAUSAL_LM"
    use_rslora: bool = False  # rank-stabilized: alpha/sqrt(r) scaling
    modules_to_save: list[str] | None = None
```

`target_modules="all-linear"` lets PEFT auto-select every linear layer. Explicit lists (`["q_proj",
"v_proj"]`) are still supported for older PEFT versions.

`use_rslora=True` (from Kalajdzievski 2023) makes higher-rank adapters trainable without
needing to sqrt-scale `lora_alpha` yourself.

## Profiles

Model-size-based defaults for LoRA and training config. Auto-selected if you don't override:

| Profile | Model size | Auto-detected? |
|---|---|---|
| `small` | ≤14B | Yes (via llm-infer `/models` or model name regex) |
| `medium` | 15–50B | Yes |
| `large` | 51–70B | Yes |
| `xlarge` | >70B | Yes |

Each profile pins `r`, `lora_alpha`, `batch_size`, `gradient_accumulation_steps`,
`max_seq_length`, `use_rslora`. Full defaults live in `training/profiles.py:MODEL_SIZE_PROFILES`.

Override at any layer:

```python
from llm_kelt.training import build_training_config

cfg = build_training_config(
    profile="medium",
    overrides={"num_epochs": 5, "learning_rate": 1e-4},
)
```

CLI override:

```bash
kelt train run manifests/x.yaml --lora-profile large
```

Detection failure raises `ProfileDetectionError` unless you pass `--lora-profile` or the
manifest sets a profile.

## Training defaults

```python
from llm_kelt.training import TRAINING_DEFAULTS

TRAINING_DEFAULTS = DotDict(
    num_epochs=3,
    batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    warmup_ratio=0.03,
    max_seq_length=2048,
    max_grad_norm=1.0,
    logging_steps=10,
    save_steps=100,
    eval_split=0.0,
    fp16=True,
    bf16=False,
    gradient_checkpointing=True,
    seed=42,
    neftune_noise_alpha=None,
)
```

Merged in this order: `TRAINING_DEFAULTS` ← profile ← manifest `training` ← CLI/keyword
overrides. Later values win.

## Adapter registry

Track versions, deploy to inference server, roll back.

```python
from llm_kelt.training import AdapterRegistry

registry = kelt.train.registry
```

### Register

```python
info = registry.register(
    training_result=result,
    key="coding-v1",
    description="First SFT pass",
    deploy=False,
    overwrite=False,
)
```

`deploy=` can be `True` / `"add"` / `"replace"`. `overwrite=True` replaces an existing
adapter of the same `key` and version — otherwise duplicate versions raise.

### List / get / remove

```python
registry.list()  # → list[AdapterInfo]
registry.get("coding-v1")  # → AdapterInfo | None
registry.remove("coding-v1")  # removes all versions
registry.remove("coding-v1", version_id="a1b2c3d4e5f6")  # single version
```

### Deploy / undeploy

```python
registry.set_deployed("coding-v1", version_id="a1b2c3d4e5f6", policy="replace")
registry.set_deployed("coding-v1", version_id=None)  # undeploy
registry.is_deployed("coding-v1")  # → bool
```

Deployment policies (`"add"` vs `"replace"`) are semantics of the inference server, not the
registry — the registry just records what should be loaded.

### Refresh the inference server

```python
registry.refresh("coding-v1", timeout=10.0)
registry.register_and_refresh(training_result=result, key="coding-v1", description="…", deploy=True)
```

`refresh` POSTs to `<infer_url>/v1/adapters/refresh` — llm-infer's endpoint for hot-reloading
adapters without restarting the model.

## Merging LoRA into base weights

For deployment to systems that don't natively load LoRA:

```bash
kelt train merge coding-v1 --model Qwen/Qwen2.5-7B-Instruct --output ./merged/ --dtype bfloat16
```

Programmatically → `llm_kelt.training.merge` (`merge_lora_adapter(...)`). Produces a full
checkpoint at `--output`. `--overwrite` replaces existing files.

## Stability detection

The runner watches training loss and gradient norms and records warnings on:

- NaN gradient norm
- Loss spike (>3σ over rolling mean)
- Divergence (loss trending up over N steps)

Warnings land in `RunResult.metrics["stability"]` and in the completed manifest, so you can
grep for unstable runs later.

## CLI shortcut summary

```bash
kelt train list                         # pending manifests
kelt train list --completed             # completed
kelt train show manifests/x.yaml
kelt train run manifests/x.yaml
kelt train sft --data f.jsonl --output ./out --epochs 3
kelt train dpo --data p.jsonl --output ./out --beta 0.1
kelt train adapters                     # registered adapters
kelt train adapters --deployed
kelt train deploy coding-v1 --policy replace
kelt train deploy coding-v1 --clear     # undeploy
kelt train merge coding-v1 --model … --output ./merged
```

## End-to-end example

[`examples/04_lora_training.py`](../examples/04_lora_training.py) trains a small LoRA adapter
against a running llm-infer instance:

1. Reads the currently-loaded model from `/v1/models`.
2. Resolves matching local HF weights via `ModelResolver`.
3. Writes a small SFT dataset to JSONL.
4. Trains with `train_lora`, `r=8`, 3 epochs.
5. Prints the adapter path and deployment instructions.

For the export-only side, [`examples/03_training_export.py`](../examples/03_training_export.py)
records synthetic feedback and preferences, then emits DPO / SFT / classifier JSONL.
