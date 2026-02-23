"""Training CLI tools - thin wrappers around training module."""

from pathlib import Path
from typing import Any

from appinfra import DotDict
from appinfra.app.tools import Tool, ToolConfig
from llm_infer import compute_adapter_metadata
from llm_infer.models import ModelResolver

from ...training.lora import AdapterRegistry
from ...training.lora import Config as LoraConfig
from ...training.manifest.loader import load_manifest
from ...training.profiles import build_training_config, get_registry_path, load_default_profile
from ...training.runner import Runner
from ...training.schema import Adapter


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

    def run(self, **kwargs: Any) -> int:
        try:
            registry_path = self._registry_path()
        except ValueError as e:
            self.lg.error(str(e))
            return 1

        subdir = "completed" if self.args.completed else "pending"
        manifests_dir = registry_path / subdir
        manifests = sorted(manifests_dir.glob("*.yaml")) if manifests_dir.exists() else []

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
        parser.add_argument("--model", "-m", help="Override model (path, HF ID, or name)")
        parser.add_argument("--skip-register", action="store_true", help="Skip registration")

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
        except (ValueError, KeyboardInterrupt, IndexError):
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

    def run(self, **kwargs: Any) -> int:
        try:
            registry_path = self._registry_path()
        except ValueError as e:
            self.lg.error(str(e))
            return 1

        manifest_path = self._get_manifest_path(registry_path)
        if manifest_path is None:
            return 0 if not self.args.manifest else 1

        try:
            runner = Runner(self.lg, registry_path, model_locations=self._model_locations())
            result = runner.run(
                manifest_path,
                skip_registration=self.args.skip_register,
                model_override=getattr(self.args, "model", None),
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
            registry = AdapterRegistry(self.lg, self._registry_path())
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

    def run(self, **kwargs: Any) -> int:
        result: int = self.group.run(**kwargs)
        return result
