# CLI reference

Entry point: `kelt`. Config is discovered via the llm-works config protocol v1: `--etc-dir` if
given, else the first-existing XDG overlay, else the wheel-bundled `llm_kelt/etc/llm-kelt.yaml`.
See [Config file lookup](#config-file-lookup) below. All subcommands accept `--help`.

Top-level structure:

```text
kelt atomic   (a)    Atomic memory maintenance
kelt proxy    (p)    OpenAI-compatible chat proxy
kelt train    (t)    Training and adapter management
kelt session  (sess) Conversation session storage
```

## `kelt atomic`

Maintenance operations on atomic memory.

### `kelt atomic vacuum`

Remove embeddings whose facts have been deleted.

```bash
kelt atomic vacuum [--dry-run]
```

`--dry-run` reports the count without deleting.

Note: currently raises `NotImplementedError` on quantized embedding stores. Tracked for a
follow-up release.

## `kelt proxy` (`p`)

Run an OpenAI-compatible HTTP server that injects atomic memory into every chat request.

### `kelt proxy serve` (`s`)

```bash
kelt proxy serve [--host HOST] [--port PORT] [--context CONTEXT_KEY]
```

| Flag | Notes |
|---|---|
| `--host`, `-h` | Bind address. Defaults from config. |
| `--port`, `-p` | Bind port. Defaults from config. |
| `--context` | `context_key` to inject facts for. Falls back to config. |

The proxy forwards `POST /v1/chat/completions` to the configured LLM backend after adding a
system message containing the assertions for `--context`. Non-chat endpoints (`/v1/models`,
etc.) pass through unmodified.

## `kelt train` (`t`)

### `kelt train run` (`r`)

Run a manifest.

```bash
kelt train run [MANIFEST | --pending] \
  [--model M] [--list-models] \
  [--skip-register] \
  [--lora-profile {small,medium,large,xlarge}]
```

Interactive picker if `MANIFEST` omitted — lists everything in `<registry>/pending/`.

With `--pending`, trains every pending manifest without prompting, oldest submission first.
Each manifest moves `pending/` → `running/` → `completed/`, or to `failed/` with the error
recorded in its `output`; a failure does not stop the rest. Exits 1 if any manifest failed,
0 otherwise (including when nothing is pending). Failed manifests are not retried; submit
again to retry.

| Flag | Notes |
|---|---|
| `--pending` | Train the whole pending queue, non-interactive. Not combined with `MANIFEST`. |
| `--model`, `-m` | Override manifest's base model (path, HF ID, or name resolvable in `models.locations`). |
| `--list-models` | Print resolvable model list and exit. |
| `--skip-register` | Train and save output but don't write to the adapter registry. |
| `--lora-profile` | Force a size profile. Otherwise auto-detected from the resolved model. |

Exit code non-zero on any failure (manifest not found, model resolution failure, training
error).

### `kelt train list` (`l`, `ls`)

```bash
kelt train list [--completed]
```

Lists pending manifests by default. `--completed` (`-c`) shows the completed directory
(supports `.yaml` and gzipped `.yaml.gz`).

### `kelt train show`

```bash
kelt train show MANIFEST
```

`MANIFEST` is a path, a name, or a bare basename (searched in the pending directory). Prints
adapter, method, model, epochs, LR, DPO beta, and data source.

### `kelt train dpo` (`d`)

Direct DPO training without a manifest.

```bash
kelt train dpo --data JSONL --output DIR \
  [--model M] [--beta B] [--no-quantize] \
  [--epochs N] [--lr LR] [--based-on PARENT]
```

Input JSONL rows: `{"prompt": ..., "chosen": ..., "rejected": ...}`.

### `kelt train sft` (`s`)

Direct SFT training.

```bash
kelt train sft --data JSONL --output DIR \
  [--model M] [--no-quantize] \
  [--epochs N] [--lr LR] [--based-on RESUME]
```

Input JSONL rows: `{"instruction"|"prompt": ..., "output"|"response": ...}`.

### `kelt train adapters` (`a`)

```bash
kelt train adapters [--deployed]
```

Lists registered adapters and whether each is currently deployed. `--deployed` (`-d`) filters
to only deployed adapters.

### `kelt train deploy` (`dp`)

```bash
kelt train deploy ADAPTER [--version V] [--policy {add,replace}] [--clear]
```

Deploy a specific version of a registered adapter, or `--clear` to remove all deployments
for the adapter. Version can be a full version ID or an md5 prefix (`abc123`), with unique
prefix resolution.

| Flag | Notes |
|---|---|
| `--version`, `-v` | Version ID or md5 prefix. Latest if omitted. |
| `--policy`, `-p` | `replace` (default) — undeploy other versions; `add` — coexist. |
| `--clear`, `-c` | Undeploy every version of `ADAPTER`. Ignores `--version`. |

### `kelt train merge` (`m`)

Bake a LoRA adapter into base model weights, producing a full checkpoint.

```bash
kelt train merge ADAPTER \
  [--model M] [--output O] \
  [--dtype {bfloat16,float16,float32}] [--overwrite]
```

`ADAPTER` is a path, a deployed name, or an md5 prefix. `--model` (`-m`) is auto-detected
from the adapter's `adapter_config.json` if omitted. Output defaults to `<model>-<adapter>`.
`--dtype` defaults to `bfloat16`. `--overwrite` skips the "output exists" prompt.

## `kelt session` (`sess`)

Manage on-disk conversation sessions written by `FileSessionStorage`. Session directory
defaults to `~/.llm-kelt/sessions`, overridable with `--sessions-dir` on `kelt session ...`.

### `kelt session list` (`ls`)

```bash
kelt session list [--limit N]
```

Prints session ID, message count, token count, and a preview of the first user message.

### `kelt session show`

```bash
kelt session show SESSION_ID [--json]
```

Human-readable dump by default. `--json` emits the full stored session as a single JSON blob.

### `kelt session delete` (`rm`)

```bash
kelt session delete SESSION_ID
```

Deletes the session file. Idempotent — no error if the session doesn't exist.

## Config file lookup

The CLI follows the [llm-works config protocol v1][protocol]. Lookup order (first match wins):

1. `--etc-dir <dir>` on the CLI → `<dir>/llm-kelt.yaml` (user-authoritative escape hatch).
2. `$XDG_CONFIG_HOME/llm-works/llm-kelt.yaml` (per-package overlay; default
   `~/.config/llm-works/llm-kelt.yaml`).
3. `$XDG_CONFIG_HOME/llm-works/config.yaml` (unified overlay, sectioned by short package
   name).
4. `$XDG_CONFIG_DIRS/llm-works/...` (system-wide defaults; default `/etc/xdg`).
5. Wheel-bundled `llm_kelt/etc/llm-kelt.yaml`.

XDG overlays typically compose the bundled base via YAML `!include`:

```yaml
# ~/.config/llm-works/llm-kelt.yaml
!include /path/to/site-packages/llm_kelt/etc/llm-kelt.yaml

dbs:
  main:
    url: postgresql://user:pass@my-host:5432/kelt   # override
```

Find the bundled base path with:

```bash
python -c "import llm_kelt, pathlib; print(pathlib.Path(llm_kelt.__file__).parent / 'etc' / 'llm-kelt.yaml')"
```

Config-field overrides via environment variables use the `INFRA_*` prefix (universal across
llm-works packages).

[protocol]:
https://github.com/llm-works/appinfra/blob/main/appinfra/docs/guides/config-protocol.md

## Registry path

`kelt train` reads its registry from `config.kelt.adapters.lora.base_path`. Without that
key, every `train` subcommand fails with a clear error message. Set it in
`etc/llm-kelt.yaml`:

```yaml
kelt:
  adapters:
    lora:
      base_path: ~/models/adapters
```

The registry directory layout:

```text
<base_path>/
  pending/           manifests waiting to run (written atomically by submit)
  running/           manifests claimed by a runner, training in progress
  completed/         manifests after successful run, with output (adapter md5, path)
  failed/            manifests whose run failed, with the error in output
  adapters/<key>/<version_id>/   adapter weights + adapter_config.json
  deployed/          symlinks to the deployed version(s)
  work/              temporary training directories
```

Producers only write `pending/`; everything else is written by the runner.
