# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-kelt Authors

"""Training CLI tools - thin wrappers around training module."""

import json
from pathlib import Path
from typing import Any

from appinfra import DotDict
from appinfra.app.tools import Tool, ToolConfig
from llm_infer import compute_adapter_metadata
from llm_infer.models import ModelResolver

from ...training.lineage import DEFAULT_SERIES, SeriesInfo
from ...training.lora import AdapterRegistry
from ...training.lora import Config as LoraConfig
from ...training.manifest.loader import load_manifest
from ...training.profiles import build_training_config, get_registry_path, load_default_profile
from ...training.runner import Runner
from ...training.runtime import load
from ...training.schema import Adapter
from ...training.storage import FileStorage, extract_md5, md5_matches


def _print_adapter_metadata(adapter_path: Path) -> None:
    """Print adapter metadata for verification."""
    meta = compute_adapter_metadata(adapter_path)
    if meta.md5 != "unknown":
        print(f"  Metadata: mtime={meta.mtime}, md5={meta.md5}")


def _print_training_result(result) -> None:
    """Print training result summary."""
    adapter_path = Path(result.adapter.path) if result.adapter else Path("unknown")
    print(f"\nTraining complete! Adapter: {adapter_path}")
    print(f"  Duration: {result.duration_seconds:.1f}s, Samples: {result.samples_trained}")
    _print_adapter_metadata(adapter_path)


class _ConfigMixin:
    """Shared config access for training tools."""

    def _config(self) -> DotDict:
        return DotDict(**dict(self.app.config)) if self.app.config else DotDict()  # type: ignore[attr-defined]

    def _registry_path(self) -> Path:
        return get_registry_path(self._config())

    def _build_config(self, method: str) -> DotDict:
        """Build training config from default profile and CLI overrides."""
        profile = load_default_profile(self._config(), method)
        overrides = {}
        if getattr(self.args, "epochs", None):  # type: ignore[attr-defined]
            overrides["num_epochs"] = self.args.epochs  # type: ignore[attr-defined]
        if getattr(self.args, "lr", None):  # type: ignore[attr-defined]
            overrides["learning_rate"] = self.args.lr  # type: ignore[attr-defined]
        return build_training_config(profile, overrides)

    def _models_config(self) -> DotDict:
        """Get models config section."""
        return DotDict(getattr(self._config(), "models", {}))

    def _print_model_locations(self, locations: list[Path]) -> None:
        """Print model search locations."""
        print("\nModel locations:")
        for loc in locations:
            marker = "" if loc.exists() else " (not found)"
            print(f"  - {loc}{marker}")

    def _print_available_models(self, locations: list[Path]) -> None:
        """Print models found in location directories."""
        print("\nAvailable models (in locations):")
        found_any = False
        for loc in locations:
            try:
                if loc.exists():
                    for model_dir in sorted(loc.iterdir()):
                        try:
                            if model_dir.is_dir() and not model_dir.name.startswith("."):
                                print(f"  - {model_dir.name}")
                                found_any = True
                        except OSError:
                            continue  # Skip unreadable entries
            except OSError:
                continue  # Skip unreadable directories
        if not found_any:
            print("  (none found)")

    def _list_models(self) -> int:
        """List available models from config."""
        models_cfg = self._models_config()
        locations = self._model_locations()

        self._print_model_locations(locations)

        # List configured models
        models = getattr(models_cfg, "models", {})
        if models:
            print("\nConfigured models:")
            for name in sorted(models.keys()):
                print(f"  - {name}")

        # Show defaults
        selection = getattr(models_cfg, "selection", None)
        if selection:
            gen = getattr(selection, "generate", None)
            emb = getattr(selection, "embed", None)
            print("\nDefaults:")
            if gen and getattr(gen, "default", None):
                print(f"  generate: {gen.default}")
            if emb and getattr(emb, "default", None):
                print(f"  embed: {emb.default}")

        self._print_available_models(locations)
        print()
        return 0

    def _model_locations(self) -> list[Path]:
        """Get model search locations from config."""
        return [Path(loc) for loc in getattr(self._models_config(), "locations", [])]

    def _get_default_model(self) -> str | None:
        """Get default model name from config."""
        models_cfg = self._models_config()
        selection = getattr(models_cfg, "selection", None)
        generate = getattr(selection, "generate", None) if selection else None
        return getattr(generate, "default", None) if generate else None

    def _resolve_model(self, model_name: str | None) -> Path | None:
        """Resolve model name to path."""
        locations = self._model_locations()
        if not locations:
            self.lg.error("no model locations configured")  # type: ignore[attr-defined]
            return None
        resolver = ModelResolver(lg=self.lg, locations=locations)  # type: ignore[attr-defined]
        name = model_name or self._get_default_model()
        if not name:
            self.lg.error("no model specified and no default configured")  # type: ignore[attr-defined]
            return None
        path = resolver.find_by_name(name)
        if path is None:
            self.lg.error("model not found", extra={"model": name})  # type: ignore[attr-defined]
        return path


class DpoTool(_ConfigMixin, Tool):
    """Train DPO adapter directly from JSONL data."""

    def __init__(self, parent: Any = None) -> None:
        super().__init__(
            parent, ToolConfig(name="dpo", aliases=["d"], help_text="Train DPO adapter")
        )

    def add_args(self, parser) -> None:
        parser.add_argument("--data", "-d", required=True, help="Input JSONL path")
        parser.add_argument("--output", "-o", required=True, help="Output adapter directory")
        parser.add_argument("--model", "-m", help="Model name")
        parser.add_argument("--beta", type=float, help="DPO beta (default: 0.1)")
        parser.add_argument("--no-quantize", action="store_true", help="Disable quantization")
        parser.add_argument("--epochs", type=int, help="Training epochs")
        parser.add_argument("--lr", type=float, help="Learning rate")
        parser.add_argument("--based-on", "-b", help="Parent adapter path")

    def run(self, **kwargs: Any) -> int:
        from ...training.dpo import train_dpo

        model_path = self._resolve_model(getattr(self.args, "model", None))
        if model_path is None:
            return 1

        config = self._build_config("dpo")
        parent = None
        if self.args.based_on:
            parent_path = Path(self.args.based_on)
            meta = compute_adapter_metadata(parent_path)
            parent = Adapter(md5=meta.md5, mtime=meta.mtime, path=str(parent_path))
        result = train_dpo(
            lg=self.lg,
            data_path=Path(self.args.data),
            output_dir=Path(self.args.output),
            base_model=str(model_path),
            lora_config=LoraConfig(),
            training_config=config,
            beta=self.args.beta or config.get("beta", 0.1),
            quantize=not self.args.no_quantize,
            parent=parent,
        )
        _print_training_result(result)
        return 0


class SftTool(_ConfigMixin, Tool):
    """Train SFT adapter directly from JSONL data."""

    def __init__(self, parent: Any = None) -> None:
        super().__init__(
            parent, ToolConfig(name="sft", aliases=["s"], help_text="Train SFT adapter")
        )

    def add_args(self, parser) -> None:
        parser.add_argument("--data", "-d", required=True, help="Input JSONL path")
        parser.add_argument("--output", "-o", required=True, help="Output adapter directory")
        parser.add_argument("--model", "-m", help="Model name")
        parser.add_argument("--no-quantize", action="store_true", help="Disable quantization")
        parser.add_argument("--epochs", type=int, help="Training epochs")
        parser.add_argument("--lr", type=float, help="Learning rate")
        parser.add_argument("--based-on", "-b", help="Resume from checkpoint")

    def run(self, **kwargs: Any) -> int:
        from ...training.lora import train_lora

        model_path = self._resolve_model(getattr(self.args, "model", None))
        if model_path is None:
            return 1

        result = train_lora(
            lg=self.lg,
            data_path=Path(self.args.data),
            output_dir=Path(self.args.output),
            base_model=str(model_path),
            lora_config=LoraConfig(),
            training_config=self._build_config("sft"),
            quantize=not self.args.no_quantize,
            resume_from=Path(self.args.based_on) if self.args.based_on else None,
        )
        _print_training_result(result)
        return 0


class ListTool(_ConfigMixin, Tool):
    """List pending training manifests."""

    def __init__(self, parent: Any = None) -> None:
        super().__init__(
            parent, ToolConfig(name="list", aliases=["l", "ls"], help_text="List manifests")
        )

    def add_args(self, parser) -> None:
        parser.add_argument("--completed", "-c", action="store_true", help="Show completed")

    def _list_manifests(self, manifests_dir: Path, completed: bool) -> list[Path]:
        """List manifest files in directory."""
        if not manifests_dir.exists():
            return []
        if completed:
            # Completed manifests can be .yaml or .yaml.gz
            return sorted(
                list(manifests_dir.glob("*.yaml")) + list(manifests_dir.glob("*.yaml.gz"))
            )
        return sorted(manifests_dir.glob("*.yaml"))

    def run(self, **kwargs: Any) -> int:
        try:
            registry_path = self._registry_path()
        except ValueError as e:
            self.lg.error(str(e))
            return 1

        subdir = "completed" if self.args.completed else "pending"
        manifests = self._list_manifests(registry_path / subdir, self.args.completed)

        if not manifests:
            print(f"No {subdir} manifests")
            return 0

        print(f"\n{subdir.capitalize()} manifests:\n")
        for path in manifests:
            try:
                m = load_manifest(path)
                records = len(m.data.records) if m.data.format == "inline" else "ext"
                print(f"  {path.name}: {m.method.upper()} {m.adapter} ({records} records)")
            except Exception as e:
                print(f"  {path.name}: (error: {e})")
        print()
        return 0


class ShowTool(_ConfigMixin, Tool):
    """Display manifest contents."""

    def __init__(self, parent: Any = None) -> None:
        super().__init__(parent, ToolConfig(name="show", help_text="Show manifest details"))

    def add_args(self, parser) -> None:
        parser.add_argument("manifest", help="Manifest path or name")

    def _resolve_path(self, ref: str) -> Path:
        path = Path(ref)
        if path.exists():
            return path
        try:
            pending = self._registry_path() / "pending"
            for candidate in [pending / ref, pending / f"{ref}.yaml"]:
                if candidate.exists():
                    return candidate
        except ValueError:
            pass
        raise FileNotFoundError(f"Manifest not found: {ref}")

    def run(self, **kwargs: Any) -> int:
        try:
            path = self._resolve_path(self.args.manifest)
            m = load_manifest(path)
        except (FileNotFoundError, ValueError) as e:
            self.lg.error(str(e))
            return 1

        print(f"\nManifest: {path.name}")
        print(f"  Adapter: {m.adapter}, Method: {m.method.upper()}")
        model = (
            m.training.get("requested_model") or m.training.get("base_model") or "(not specified)"
        )
        print(f"  Model: {model}")
        print(
            f"  Epochs: {m.training.get('num_epochs', 3)}, LR: {m.training.get('learning_rate', 2e-4)}"
        )
        if m.method == "dpo":
            print(f"  Beta: {m.method_config.get('beta', 0.1)}")
        records = len(m.data.records) if m.data.format == "inline" else m.data.path
        print(f"  Data: {m.data.format} ({records})\n")
        return 0


class RunTool(_ConfigMixin, Tool):
    """Run training from manifest."""

    def __init__(self, parent: Any = None) -> None:
        super().__init__(
            parent, ToolConfig(name="run", aliases=["r"], help_text="Run manifest training")
        )

    def add_args(self, parser) -> None:
        parser.add_argument("manifest", nargs="?", help="Manifest path (interactive if omitted)")
        parser.add_argument(
            "--pending",
            action="store_true",
            help="Train all pending manifests, non-interactive; exit 1 if any failed",
        )
        parser.add_argument("--model", "-m", help="Override model (path, HF ID, or name)")
        parser.add_argument(
            "--list-models", action="store_true", help="List available models and exit"
        )
        parser.add_argument("--skip-register", action="store_true", help="Skip registration")
        parser.add_argument(
            "--lora-profile",
            choices=["small", "medium", "large", "xlarge"],
            help="LoRA/training profile (auto-detected from model size if omitted)",
        )

    def _select_interactive(self, pending_dir: Path) -> Path | None:
        manifests = sorted(pending_dir.glob("*.yaml"))
        if not manifests:
            print("No pending manifests")
            return None

        print("\nPending manifests:\n")
        for i, path in enumerate(manifests, 1):
            try:
                m = load_manifest(path)
                records = len(m.data.records) if m.data.format == "inline" else "ext"
                print(f"  [{i}] {path.name}: {m.method.upper()} {m.adapter} ({records})")
            except Exception:
                print(f"  [{i}] {path.name}: (error)")

        try:
            choice = input("\nSelect (number or 'q'): ").strip()
            if choice.lower() == "q":
                return None
            idx = int(choice) - 1
            return manifests[idx] if 0 <= idx < len(manifests) else None
        except (ValueError, KeyboardInterrupt, IndexError, EOFError):
            return None

    def _get_manifest_path(self, registry_path: Path) -> Path | None:
        """Get manifest path from args or interactive selection."""
        if self.args.manifest:
            path = Path(self.args.manifest)
            if not path.exists():
                self.lg.error(f"Not found: {path}")
                return None
            return path
        return self._select_interactive(registry_path / "pending")

    def _run_pending(self, registry_path: Path) -> int:
        """Train all pending manifests; 1 if any failed, 0 otherwise (also when none)."""
        storage = FileStorage(self.lg, registry_path)
        runner = Runner(self.lg, storage, model_locations=self._model_locations())
        results = runner.run_pending(
            skip_registration=self.args.skip_register,
            model_override=getattr(self.args, "model", None),
            lora_profile=getattr(self.args, "lora_profile", None),
        )
        if not results:
            print("No pending manifests")
            return 0

        failed = 0
        for adapter, result in results:
            if result.status == "failed":
                failed += 1
                print(f"  FAILED  {adapter}: {result.error}")
            else:
                md5 = result.adapter.md5 if result.adapter else "unknown"
                print(f"  OK      {adapter}: adapter {md5[:12]} ({result.duration_seconds:.0f}s)")
        print(f"\n{len(results) - failed} trained, {failed} failed")
        return 1 if failed else 0

    def run(self, **kwargs: Any) -> int:
        if self.args.list_models:
            return self._list_models()

        try:
            registry_path = self._registry_path()
        except ValueError as e:
            self.lg.error(str(e))
            return 1

        if self.args.pending:
            return self._run_queue(registry_path)
        return self._run_manifest(registry_path)

    def _run_queue(self, registry_path: Path) -> int:
        """Handle --pending: train the queue, reporting errors that stop the whole batch."""
        if self.args.manifest:
            self.lg.error("--pending trains the queue; do not pass a manifest path")
            return 1
        try:
            return self._run_pending(registry_path)
        except Exception as e:
            self.lg.error(f"Training queue failed: {e}")
            return 1

    def _run_manifest(self, registry_path: Path) -> int:
        """Train one manifest, given as a path or picked interactively from pending/."""
        manifest_path = self._get_manifest_path(registry_path)
        if manifest_path is None:
            return 0 if not self.args.manifest else 1

        try:
            storage = FileStorage(self.lg, registry_path)
            runner = Runner(self.lg, storage, model_locations=self._model_locations())
            result = runner.run(
                manifest_path,
                skip_registration=self.args.skip_register,
                model_override=getattr(self.args, "model", None),
                lora_profile=getattr(self.args, "lora_profile", None),
            )
        except Exception as e:
            self.lg.error(f"Training failed: {e}")
            return 1

        self._print_manifest_result(result)
        return 0

    def _print_manifest_result(self, result) -> None:
        """Print manifest training result."""
        adapter_path = Path(result.adapter.path) if result.adapter else Path("unknown")
        md5 = result.adapter.md5 if result.adapter else "unknown"
        print(f"\nTraining complete! Adapter: {md5[:12]}")
        print(f"  Path: {adapter_path}, Duration: {result.duration_seconds:.1f}s")
        _print_adapter_metadata(adapter_path)


class DeployTool(_ConfigMixin, Tool):
    """Deploy an adapter version."""

    def __init__(self, parent: Any = None) -> None:
        super().__init__(
            parent, ToolConfig(name="deploy", aliases=["dp"], help_text="Deploy adapter version")
        )

    def add_args(self, parser) -> None:
        parser.add_argument("adapter", help="Adapter key (e.g., jokester-p-sft)")
        parser.add_argument("--version", "-v", help="Version ID or md5 prefix (latest if omitted)")
        parser.add_argument(
            "--policy",
            "-p",
            choices=["add", "replace"],
            default="replace",
            help="Deployment policy (default: replace)",
        )
        parser.add_argument(
            "--clear", "-c", action="store_true", help="Remove all deployments for adapter"
        )
        parser.add_argument("--series", "-s", help="Deploy a build of this series")
        parser.add_argument("--commit", help="Commit of --series (default: its current build)")

    def _resolve_version(
        self, storage: FileStorage, adapter: str, version: str | None
    ) -> str | None:
        """Resolve version argument to full version_id."""
        if version is None:
            return None

        version_ids = storage.list_versions(adapter)
        if not version_ids:
            raise ValueError(f"No versions found for adapter '{adapter}'")

        # Try exact match on version_id or md5
        for vid in version_ids:
            if vid == version or extract_md5(vid) == version.lower():
                return vid

        # Try pattern match (supports prefix, suffix, and prefix..suffix)
        matches = [vid for vid in version_ids if md5_matches(vid, version)]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(f"Ambiguous md5 pattern '{version}': {matches}")

        raise ValueError(f"Version '{version}' not found for adapter '{adapter}'")

    def _deploy(self, storage: FileStorage, adapter: str) -> None:
        """Deploy a series build (--series) or a version (--version, default: latest)."""
        if self.args.series:
            if self.args.version:
                raise ValueError("Give --version or --series, not both")
            storage.deploy_series(
                adapter, self.args.series, self.args.commit, policy=self.args.policy
            )
            return
        version_id = self._resolve_version(storage, adapter, self.args.version)
        storage.deploy_adapter(adapter, version_id, policy=self.args.policy)

    def _print_deployed(self, storage: FileStorage, adapter: str) -> None:
        """Print currently deployed versions."""
        deployed = storage.list_deployed(adapter)
        if deployed:
            print(f"\nDeployed {adapter}:")
            for key, md5 in deployed:
                print(f"  {key}-{md5}")
        print()

    def run(self, **kwargs: Any) -> int:
        try:
            storage = FileStorage(self.lg, self._registry_path())
        except ValueError as e:
            self.lg.error(str(e))
            return 1

        adapter = self.args.adapter
        try:
            if self.args.clear:
                storage.undeploy_adapter(adapter)
                print(f"\nCleared all deployments for {adapter}")
                return 0

            self._deploy(storage, adapter)
            self._print_deployed(storage, adapter)
            return 0
        except ValueError as e:
            self.lg.error(str(e))
            return 1
        except Exception as e:
            self.lg.error(f"{e.__class__.__name__}: {e}")
            return 1


class AdaptersTool(_ConfigMixin, Tool):
    """List registered adapters."""

    def __init__(self, parent: Any = None) -> None:
        super().__init__(
            parent, ToolConfig(name="adapters", aliases=["a"], help_text="List adapters")
        )

    def add_args(self, parser) -> None:
        parser.add_argument("--deployed", "-d", action="store_true", help="Show only deployed")

    def run(self, **kwargs: Any) -> int:
        try:
            storage = FileStorage(self.lg, self._registry_path())
            registry = AdapterRegistry(self.lg, storage)
        except ValueError as e:
            self.lg.error(str(e))
            return 1

        adapters = registry.list()
        if self.args.deployed:
            adapters = [a for a in adapters if a.deployed]

        if not adapters:
            print("No adapters")
            return 0

        print("\nAdapters:\n")
        for a in sorted(adapters, key=lambda x: x.key):
            status = "[deployed]" if a.deployed else ""
            print(f"  {a.key} {status}")
            if a.description:
                print(f"    {a.description}")
        print()
        return 0


class SeriesTool(_ConfigMixin, Tool):
    """Show, fork, annotate, prune or drop a training series; find a set's or version's."""

    def __init__(self, parent: Any = None) -> None:
        super().__init__(parent, ToolConfig(name="series", help_text="Show a training series"))

    def add_args(self, parser) -> None:
        parser.add_argument("adapter", nargs="?", help="Adapter key")
        parser.add_argument("--series", "-s", default=DEFAULT_SERIES, help="Series name")
        parser.add_argument("--set", help="Find the series containing this set id")
        parser.add_argument("--md5", help="Show the series and commit of an adapter version")
        parser.add_argument("--fork", metavar="NEW", help="Start series NEW at --commit")
        parser.add_argument("--commit", help="Commit to fork at (default: the series' tip)")
        parser.add_argument(
            "--annotate", metavar="JSON", help="Merge a JSON object into --commit's build or --set"
        )
        parser.add_argument(
            "--prune", action="store_true", help="Delete the series' undeployed weights"
        )
        parser.add_argument(
            "--drop", action="store_true", help="Delete the series with its sets and weights"
        )
        parser.add_argument(
            "--wipe", action="store_true", help="Delete everything of the adapter key"
        )

    def run(self, **kwargs: Any) -> int:
        try:
            storage = FileStorage(self.lg, self._registry_path())
            if self.args.annotate:
                return self._annotate(storage, json.loads(self.args.annotate))
            if self.args.set:
                for adapter, series in storage.lineage.find_set(self.args.set):
                    print(f"{adapter}/{series}")
                return 0
            if self.args.md5:
                return self._print_version(storage, self.args.md5)
            if not self.args.adapter:
                raise ValueError("Give an adapter key, --set or --md5")
            return self._run_on_adapter(storage)
        except (ValueError, json.JSONDecodeError) as e:
            self.lg.error(str(e))
            return 1

    def _run_on_adapter(self, storage: FileStorage) -> int:
        adapter, series = self.args.adapter, self.args.series
        if self.args.fork:
            storage.validate_key(self.args.fork)
            commit = storage.lineage.fork(adapter, series, self.args.fork, self.args.commit)
            print(f"Started {adapter}/{self.args.fork} at {commit[:12]}")
        elif self.args.prune:
            pruned = storage.prune_series_adapters(adapter, series)
            print(f"Pruned the builds of {len(pruned)} commits of {adapter}/{series}")
        elif self.args.drop:
            dropped = storage.drop_series(adapter, series)
            print(f"Dropped {adapter}/{series} ({len(dropped)} commits only it held)")
        elif self.args.wipe:
            storage.wipe_adapter(adapter)
            print(f"Wiped {adapter}")
        else:
            return self._show(storage)
        return 0

    def _annotate(self, storage: FileStorage, annotations: Any) -> int:
        if not self.args.adapter or bool(self.args.commit) == bool(self.args.set):
            raise ValueError("--annotate needs an adapter key and one of --commit or --set")
        if self.args.commit:
            storage.lineage.annotate_build(self.args.adapter, self.args.commit, annotations)
        else:
            storage.lineage.annotate_set(self.args.adapter, self.args.set, annotations)
        return 0

    def _show(self, storage: FileStorage) -> int:
        info = storage.lineage.series(self.args.adapter, self.args.series)
        if info is None:
            print(f"No series {self.args.adapter}/{self.args.series}")
            return 1
        active = storage.active_series(self.args.adapter)
        self._print_series(info, active.commit if active and active.series == info.series else None)
        return 0

    def _print_version(self, storage: FileStorage, md5: str) -> int:
        manifest = storage.find_adapter_by_md5(md5)
        if manifest is None:
            print(f"No completed training for adapter version {md5}")
            return 1
        if not manifest.series:
            print(f"{manifest.adapter}: one-off training, not part of a series")
            return 0
        print(f"{manifest.adapter}/{manifest.series} at commit {manifest.commit}")
        return 0

    def _print_series(self, info: SeriesInfo, deployed: str | None) -> None:
        current = info.current_build
        print(f"\nSeries {info.adapter}/{info.series}: {info.recipe['method'].upper()}")
        print(f"  Base model: {info.base_model}")
        print(f"  Tip: {info.tip}, current build: {current.commit if current else '-'}")
        print(f"  Active: {'yes, deployed ' + deployed if deployed else 'no'}")
        print("\n  Sets (oldest first):")
        for entry in info.sets:
            date = entry.submitted_at.isoformat(timespec="seconds") if entry.submitted_at else "-"
            build = entry.build.state if entry.build else "not built"
            print(f"    {entry.commit}  set {entry.set_id}  {date}  {build}")
            if entry.annotations:
                print(f"      set: {json.dumps(entry.annotations)}")
            if entry.build and entry.build.annotations:
                print(f"      build: {json.dumps(entry.build.annotations)}")
        print()


class MergeTool(_ConfigMixin, Tool):
    """Merge LoRA adapter into base model weights."""

    def __init__(self, parent: Any = None) -> None:
        super().__init__(
            parent, ToolConfig(name="merge", aliases=["m"], help_text="Merge adapter into model")
        )

    def add_args(self, parser) -> None:
        parser.add_argument("adapter", help="Adapter path, deployed name, or md5 hash")
        parser.add_argument("--model", "-m", help="Base model name (auto-detected if omitted)")
        parser.add_argument("--output", "-o", help="Output path (default: <model>-<adapter>)")
        parser.add_argument(
            "--dtype",
            choices=["bfloat16", "float16", "float32"],
            default="bfloat16",
            help="Output dtype (default: bfloat16)",
        )
        parser.add_argument(
            "--overwrite", action="store_true", help="Overwrite existing output without prompting"
        )

    def _find_by_md5(self, storage: FileStorage, pattern: str) -> Path | None:
        """Search all adapters for a version matching md5 pattern."""
        matches: list[tuple[str, str]] = []  # (key, version_id)
        for key in storage.list_adapters():
            for vid in storage.list_versions(key):
                if md5_matches(vid, pattern):
                    matches.append((key, vid))
        if not matches:
            return None
        if len(matches) > 1:
            match_strs = [f"{k}/{v}" for k, v in matches]
            self.lg.error(
                "ambiguous md5 pattern", extra={"pattern": pattern, "matches": match_strs}
            )
            return None
        key, vid = matches[0]
        return storage.get_adapter_path(key, vid)

    def _resolve_adapter_path(self, adapter: str) -> Path | None:
        """Resolve adapter argument to path, deployed name, or md5."""
        adapter_path = Path(adapter)
        if adapter_path.exists():
            return adapter_path
        storage = FileStorage(self.lg, self._registry_path())
        # Try deployed name first - iterate all entries to find valid match
        for key, md5 in storage.list_deployed(adapter):
            for vid in storage.list_versions(key):
                if md5_matches(vid, md5):
                    return storage.get_adapter_path(key, vid)
        # Try md5 lookup (search all adapters)
        if path := self._find_by_md5(storage, adapter):
            return path
        self.lg.error("adapter not found", extra={"adapter": adapter})
        return None

    def _get_base_model_from_adapter(self, adapter_path: Path) -> str | None:
        """Extract base model path from adapter config."""
        import json

        config_path = adapter_path / "adapter_config.json"
        if not config_path.exists():
            return None
        with open(config_path) as f:
            result: str | None = json.load(f).get("base_model_name_or_path")
            return result

    def _resolve_base_model(self, adapter_path: Path) -> Path | None:
        """Resolve base model from args or adapter config."""
        if self.args.model:
            return self._resolve_model(self.args.model)
        base_model = self._get_base_model_from_adapter(adapter_path)
        if not base_model:
            self.lg.error("could not detect base model, use --model")
            return None
        model_path = Path(base_model)
        return model_path if model_path.exists() else self._resolve_model(model_path.name)

    def _extract_md5_suffix(self, adapter_path: Path) -> str:
        """Extract md5 suffix from adapter path (format: YYYYMMDD-HHMMSS-md5)."""
        name = adapter_path.name if adapter_path.is_dir() else adapter_path.stem
        return extract_md5(name)

    def _get_output_path(self, model_path: Path, adapter_path: Path) -> Path | None:
        """Determine output path, prompting for overwrite if exists."""
        if self.args.output:
            output_path = Path(self.args.output).resolve()
        else:
            md5_suffix = self._extract_md5_suffix(adapter_path)
            output_path = (model_path.parent / f"{model_path.name}-{md5_suffix}").resolve()
        # Safety check: ensure output doesn't overlap with source paths
        model_resolved = model_path.resolve()
        adapter_resolved = adapter_path.resolve()
        if output_path == model_resolved or output_path == adapter_resolved:
            self.lg.error("output path cannot be same as model or adapter path")
            return None
        try:
            if model_resolved.is_relative_to(output_path) or adapter_resolved.is_relative_to(
                output_path
            ):
                self.lg.error("output path cannot be ancestor of model or adapter path")
                return None
        except ValueError:
            pass  # Not relative, which is fine
        if output_path.exists() and not self.args.overwrite:
            response = input(f"Output path exists: {output_path}\nOverwrite? [y/N] ")
            if response.lower() not in ("y", "yes"):
                self.lg.info("aborted by user")
                return None
        return output_path

    def _is_visual_key(self, key: str) -> bool:
        """Check if a weight key belongs to the visual tower."""
        return ".visual." in key or key.startswith("visual.")

    def _load_visual_weights(self, model_path: Path) -> dict[str, Any]:
        """Load visual tower weights from base model safetensors."""
        import json

        from safetensors.torch import load_file

        weights: dict[str, Any] = {}
        index_path = model_path / "model.safetensors.index.json"
        if index_path.exists():
            # Sharded checkpoint: load only files containing visual weights
            with open(index_path) as f:
                weight_map = json.load(f).get("weight_map", {})
            visual_files = {f for k, f in weight_map.items() if self._is_visual_key(k)}
            for filename in visual_files:
                for key, tensor in load_file(model_path / filename).items():
                    if self._is_visual_key(key):
                        weights[key] = tensor
        else:
            # Non-sharded checkpoint: single model.safetensors file
            single_file = model_path / "model.safetensors"
            if single_file.exists():
                for key, tensor in load_file(single_file).items():
                    if self._is_visual_key(key):
                        weights[key] = tensor
        return weights

    def _copy_visual_weights(self, model_path: Path, output_path: Path) -> None:
        """Copy visual tower weights from base model (for VLM architectures)."""
        import json

        from safetensors.torch import save_file

        visual_weights = self._load_visual_weights(model_path)
        if not visual_weights:
            return
        save_file(visual_weights, output_path / "visual.safetensors")
        self.lg.info("copied visual tower weights", extra={"count": len(visual_weights)})
        # Update or create output index to include visual weights
        out_index_path = output_path / "model.safetensors.index.json"
        if out_index_path.exists():
            with open(out_index_path) as f:
                out_index = json.load(f)
        else:
            out_index = {"metadata": {}, "weight_map": {}}
        for key in visual_weights:
            out_index["weight_map"][key] = "visual.safetensors"
        with open(out_index_path, "w") as f:
            json.dump(out_index, f, indent=2)

    def _load_and_merge(self, model_path: Path, adapter_path: Path, dtype: Any) -> Any:
        """Load base model, apply adapter, and return merged model."""
        from peft import PeftModel
        from transformers import AutoModelForCausalLM

        self.lg.info("loading base model", extra={"model": str(model_path)})
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            torch_dtype=dtype,
            device_map="cpu",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        self.lg.info("loading adapter", extra={"adapter": str(adapter_path)})
        peft_model = PeftModel.from_pretrained(model, str(adapter_path))
        self.lg.info("merging adapter into base model")
        return peft_model.merge_and_unload()  # type: ignore[operator]

    def _is_bnb_model_path(self, model_path: Path) -> bool:
        """Check if model path contains BNB quantized weights."""
        config_path = model_path / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            quant_config = config.get("quantization_config", {})
            return bool(quant_config.get("quant_method") == "bitsandbytes")
        return False

    def _is_sharded_model(self, model_path: Path) -> bool:
        """Check if model uses sharded safetensors (has index file)."""
        return (model_path / "model.safetensors.index.json").exists()

    def _copy_configs_and_tokenizer(self, model_path: Path, output_path: Path) -> None:
        """Copy config files and tokenizer from base model."""
        import shutil

        from transformers import AutoTokenizer

        for cfg in model_path.glob("*.json"):
            if "index" not in cfg.name:
                shutil.copy(cfg, output_path / cfg.name)
        self.lg.info("restored base model configs")
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
        tokenizer.save_pretrained(output_path)
        self.lg.info("copied tokenizer")

    def _replace_safetensors(self, target: Path, source: Path) -> None:
        """Replace safetensor files in target with those from source."""
        import shutil

        for sf in target.glob("model*.safetensors"):
            sf.unlink()
        (target / "model.safetensors.index.json").unlink(missing_ok=True)
        for sf in source.glob("*.safetensors"):
            shutil.move(str(sf), target / sf.name)
        idx = source / "model.safetensors.index.json"
        if idx.exists():
            shutil.move(str(idx), target / idx.name)

    def _requantize_bnb(self, model_path: Path) -> None:
        """Re-quantize a fp16 model to BNB 4-bit in place using layerwise quantization."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp) / "model"
            load("merge").quantize_layerwise(self.lg, model_path, tmp_path)
            self._replace_safetensors(model_path, tmp_path)

    def _do_merge(
        self, model_path: Path, adapter_path: Path, output_path: Path, dtype: Any
    ) -> bool:
        """Merge adapter into model. Returns True if layerwise merge was used."""
        # Use layerwise merge only for sharded BNB models (non-sharded fall back to standard)
        use_layerwise = self._is_bnb_model_path(model_path) and self._is_sharded_model(model_path)

        if use_layerwise:
            self.lg.info("using layerwise merge for BNB quantized model")
            load("merge").merge_layerwise(self.lg, model_path, adapter_path, output_path, dtype)
        else:
            merged = self._load_and_merge(model_path, adapter_path, dtype)
            self.lg.info("saving merged model", extra={"output": str(output_path)})
            merged.save_pretrained(output_path)

        return use_layerwise

    def _merge_and_save(self, model_path: Path, adapter_path: Path, output_path: Path) -> None:
        """Load model, merge adapter, and save result."""
        import shutil

        import torch

        dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
        dtype = dtype_map[self.args.dtype]

        if output_path.exists():
            shutil.rmtree(output_path)

        use_layerwise = self._do_merge(model_path, adapter_path, output_path, dtype)
        self._copy_configs_and_tokenizer(model_path, output_path)

        if use_layerwise:
            self._requantize_bnb(output_path)

        # Copy visual weights after requantization so index entries are preserved
        self._copy_visual_weights(model_path, output_path)

    def run(self, **kwargs: Any) -> int:
        adapter_path = self._resolve_adapter_path(self.args.adapter)
        if adapter_path is None:
            return 1
        model_path = self._resolve_base_model(adapter_path)
        if model_path is None:
            return 1
        output_path = self._get_output_path(model_path, adapter_path)
        if output_path is None:
            return 1
        try:
            self._merge_and_save(model_path, adapter_path, output_path)
        except Exception as e:
            self.lg.error("merge failed", extra={"exception": e})
            return 1
        print(f"\nMerged model saved to: {output_path}")
        return 0


class TrainTool(Tool):
    """Training commands."""

    def __init__(self, parent: Any = None) -> None:
        super().__init__(
            parent, ToolConfig(name="train", aliases=["t"], help_text="Training commands")
        )
        self.add_tool(RunTool(self))
        self.add_tool(ListTool(self))
        self.add_tool(ShowTool(self))
        self.add_tool(DpoTool(self))
        self.add_tool(SftTool(self))
        self.add_tool(AdaptersTool(self))
        self.add_tool(DeployTool(self))
        self.add_tool(SeriesTool(self))
        self.add_tool(MergeTool(self))

    def run(self, **kwargs: Any) -> int:
        result: int = self.group.run(**kwargs)
        return result
