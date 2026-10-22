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

### Train the queue

`kelt train run --pending` (or `runner.run_pending()`) trains every manifest in
`<registry>/pending/`, oldest submission first. Each one is claimed by an atomic move to
`running/`, then lands in `completed/` or, on any error, in `failed/` with
`output.status: failed` and `output.error`. One failure does not stop the batch; the
command exits 1 if any manifest failed. Failed manifests stay in `failed/` until resubmitted.

A producer submits with `kelt.train.manifest.submit(manifest)` and polls the result:

```python
status = kelt.train.manifest.status("coding-v1")  # → QueueStatus | None
status.state  # "pending" | "running" | "completed" | "failed"
status.created_at  # identifies the submission
status.output.adapter.md5  # completed: the trained adapter's version id
status.output.error  # failed: the reason
```

`status()` reads manifest metadata only, never the training data. A manifest still queued
for the key wins over finished runs; among finished runs the newest wins. A manifest left in
`running/` by a killed process stays there until removed by hand.

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

### Series: training sets that accumulate

A manifest with `series` set holds a new training *set* of that series, not a full training
job. Each build of a series trains from the base model on every set submitted to it, oldest
first, and writes a new version of the adapter; no weights carry over between builds.

```python
manifest = kelt.train.manifest.create(
    adapter="coding-v1",
    method="sft",
    model="Qwen/Qwen2.5-7B-Instruct",
    data=todays_records,
    series="main",
)
result = kelt.train.manifest.submit(manifest)
result.commit  # the lineage commit holding this set
result.created  # False: the set was already in the series, nothing was added
result.root  # True: this set started the series
```

- The history works like git. The adapter key is a repository in `<registry>/cas/` and a
  series is a branch of it. Each set becomes a commit, identified by the hash of its records
  (in any order) and templates. A set is in a series' history at most once, so resubmitting
  it adds nothing.
- A series' first submit fixes its base model and recipe (method, `lora`, `training`, method
  config). A later submit with different ones is rejected; start a new series to change them.
  The runner rejects `--model` and `--lora-profile` overrides for series builds.
- A series has at most one pending build, of its tip: the runner trains whatever the tip is
  when it claims the build. Resubmitting a set re-queues the tip if its last build failed.
  Series `main` queues under the adapter key; another series under `<adapter>+<series>`
  (`status(adapter, series=...)`), so series of one adapter build independently.
- Only series `main` deploys when a build completes; other series need deployment policy
  `skip` and are deployed by commit (see below). `parent` is not supported on series
  manifests.
- `kelt.train.manifest.fork(adapter, series, new_series, commit=None)` starts a series at a
  commit of another (default: its tip), sharing that history and its builds.
- Completed series manifests record the `commit` they trained and keep no records; the sets
  stay in the store.

```python
info = kelt.train.manifest.series("coding-v1")  # → SeriesInfo | None
info.tip, info.base_model, info.recipe
info.current_build  # newest SetEntry with a completed build
for entry in info.sets:  # oldest first
    entry.commit, entry.set_id, entry.submitted_at, entry.annotations
    entry.build  # state (completed, failed, pruned), md5, error, annotations
kelt.train.manifest.find_set(set_id)  # → [(adapter, series), ...]
```

An adapter version's series and commit are in its `config.yaml` (`series`, `commit`) and in
its completed manifest: `kelt.train.manifest.get_manifest(md5).series` / `.commit`.

#### Starting over and deploying a series

Starting over from the base model, with a new recipe or on new data only, is a new series of
the same adapter key: submit sets with `series="v2"`. The key keeps serving its deployed
version while the new series trains; deploying one of its builds switches the key over.

```python
kelt.train.manifest.deploy("coding-v1", "v2", commit=None)  # default: v2's current build
kelt.train.manifest.active("coding-v1")  # → ActiveSeries(series, commit, version_id, md5)
```

- `deploy` replaces the key's deployed versions; it links the new version before removing
  the old ones, so the key is never left undeployed.
- The active series is the series of the deployed version. "Latest version" (`deploy`
  without a version, `get_adapter`) considers versions of `main` and of the active series.
- An older series stays as history: its sets, build records and weights remain.

#### Annotations

Sets and builds carry annotations: small JSON objects (at most 64 KiB each), mutable, kept
outside the content-addressed store so they never change an id.

```python
kelt.train.manifest.create(..., series="v2", annotations={"prompt": "v3"})  # the set's
kelt.train.manifest.annotate_set("coding-v1", set_id, {"note": "use with caution"})
kelt.train.manifest.annotate_build("coding-v1", commit, {"gate": "passed", "eval_loss": 0.41})
```

Annotations merge key by key. A set's start as those of its first submit. A build's survive
pruning its weights and are reset when the commit is rebuilt.

#### Pruning

- `prune(adapter, series)` deletes the weights of the series' builds except deployed ones.
  The sets stay; each build stays recorded as `pruned` with its md5 and annotations, and a
  resubmit of a pruned tip rebuilds it. Builds a series shares with a fork are pruned for
  both.
- `drop(adapter, series)` deletes the series with the sets, builds, weights and annotations
  only it holds; its training data is gone afterwards. Refused while the series is deployed
  or a build of it runs.
- `wipe(adapter)` deletes everything of the key: series, sets, builds, weights and
  deployments. Completed and failed manifests stay.

The CLI covers the same: `kelt train series coding-v1 [--series main]`,
`kelt train series --set <set id>`, `kelt train series --md5 <md5>`,
`kelt train series coding-v1 --fork trial [--commit <commit>]`,
`kelt train series coding-v1 --annotate '<json>' (--commit <commit> | --set <set id>)`,
`kelt train series coding-v1 -s v1 --prune | --drop`, `kelt train series coding-v1 --wipe`
and `kelt train deploy coding-v1 --series v2 [--commit <commit>]`.

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

## Supported models

SFT (LoRA), DPO and prompt tuning are tested on these model families:

| Family | `model_type` | Default LoRA targets |
|---|---|---|
| Qwen2.5 | `qwen2` | `q/k/v/o_proj`, `gate/up/down_proj` |
| Qwen3.5, Qwen3.8 | `qwen3_5` | the above plus `in_proj_qkv`, `in_proj_z`, `out_proj` (linear-attention layers); vision tower and multi-token-prediction head excluded |

Other families fall back to the Qwen2.5 targets; set `target_modules` for them.

The model is loaded with the class its checkpoint was saved with (`config.architectures`), so a
multimodal checkpoint such as `Qwen3_5ForConditionalGeneration` keeps its layer names; the adapter
then matches what an inference server loads for that checkpoint. Training uses text only.

Checkpoint formats:

| Format | How kelt loads it |
|---|---|
| bf16 / fp16 | quantized to 4-bit with bitsandbytes (QLoRA) unless `quantize=False` |
| bitsandbytes 4-bit | as stored |
| compressed-tensors (e.g. llm-compressor W4A16) | dequantized to the model dtype at load; LoRA trains on dense weights, so memory is that of the unquantized model. Needs the `compressed-tensors` package and a local checkpoint path |

GPTQ and AWQ checkpoints are not supported for training. `kelt train merge` supports
bitsandbytes 4-bit bases only.

## LoRA config

```python
class Config:  # llm_kelt.training.lora.Config
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] | None = None  # None = model family default
    exclude_modules: str | None = None  # regex; None = model family default
    bias: Literal["none", "all", "lora_only"] = "none"
    task_type: str = "CAUSAL_LM"
    use_rslora: bool = False  # rank-stabilized: alpha/sqrt(r) scaling
```

Unset `target_modules` / `exclude_modules` are filled in from the loaded model's `model_type`
(see Supported models); the resolved list is recorded in the run's `config.lora`.

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

> **Note:** On Ampere+ GPUs (RTX 30xx, A100, etc.), models typically load in bf16. Pass
> `bf16=True, fp16=False` to keep trainer precision aligned — the default `fp16=True` can
> trigger a grad-scaler error when the model weights are bf16.

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
kelt train deploy coding-v1 --series v2 # deploy series v2's current build
kelt train series coding-v1             # sets and builds of series main
kelt train series --set <set id>        # series containing a set
kelt train series --md5 <md5>           # series and commit of an adapter version
kelt train series coding-v1 --fork trial
kelt train merge coding-v1 --model … --output ./merged
```

## End-to-end example

[`examples/lora_training.py`](../examples/lora_training.py) trains a small LoRA adapter
against a running llm-infer instance:

1. Reads the currently-loaded model from `{base_url}/models` (the default config's
   `base_url` ends in `/v1`, so this hits `/v1/models`).
2. Resolves matching local HF weights via `ModelResolver`.
3. Writes a small SFT dataset to JSONL.
4. Trains with `train_lora`, `r=8`, 3 epochs.
5. Prints the adapter path and deployment instructions.

For the export-only side, [`examples/training_export.py`](../examples/training_export.py)
records synthetic feedback and preferences, then emits DPO / SFT / classifier JSONL.
