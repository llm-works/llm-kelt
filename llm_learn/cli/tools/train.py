"""Training tools - CLI for DPO training workflow."""

from datetime import datetime
from pathlib import Path
from typing import Any

from appinfra.app.tools import Tool, ToolConfig
from appinfra.db.pg import PG
from appinfra.dot_dict import DotDict
from llm_infer.models import ModelResolver

from ...core.database import Database
from ...training.config import LoraConfig, TrainingConfig, TrainingResult


class ModelResolutionMixin:
    """Mixin for tools that need model resolution."""

    def _get_models_config(self) -> DotDict:
        """Get models config section."""
        config = DotDict(**dict(self.app.config)) if self.app.config else DotDict()  # type: ignore[attr-defined]
        return getattr(config, "models", DotDict())

    def _get_model_locations(self) -> list[Path]:
        """Get model search locations from config."""
        models_cfg = self._get_models_config()
        locations = getattr(models_cfg, "locations", [])
        if not locations:
            return []
        return [Path(loc) for loc in locations]

    def _get_selection_default(self) -> str | None:
        """Get default model from selection.generate.default."""
        models_cfg = self._get_models_config()
        selection = getattr(models_cfg, "selection", None)
        if selection is None:
            return None
        generate = getattr(selection, "generate", None)
        if generate is None:
            return None
        return getattr(generate, "default", None)

    def _resolve_model(self, model_name: str | None) -> Path | None:
        """Resolve model name to path using ModelResolver."""
        locations = self._get_model_locations()
        if not locations:
            self.lg.error("no model locations configured in models.locations")  # type: ignore[attr-defined]
            return None

        resolver = ModelResolver(lg=self.lg, locations=locations)  # type: ignore[attr-defined]

        # If model specified, resolve it
        if model_name:
            path = resolver.find_by_name(model_name)
            if path is None:
                self.lg.error(  # type: ignore[attr-defined]
                    "model not found",
                    extra={"model": model_name, "locations": [str(p) for p in locations]},
                )
            return path

        # Fall back to selection default
        default = self._get_selection_default()
        if default:
            path = resolver.find_by_name(default)
            if path is None:
                self.lg.error(  # type: ignore[attr-defined]
                    "default model not found",
                    extra={"model": default, "locations": [str(p) for p in locations]},
                )
            return path

        self.lg.error("no model specified and no default configured")  # type: ignore[attr-defined]
        return None

    def _list_models(self) -> int:
        """List available models from config."""
        models_cfg = self._get_models_config()
        models = getattr(models_cfg, "models", {})
        default = self._get_selection_default()

        if not models:
            print("No models configured in models.models")
            return 0

        print("Available models:\n")
        for name in sorted(models.keys() if hasattr(models, "keys") else []):
            marker = " (default)" if name == default else ""
            print(f"  - {name}{marker}")
        print()
        return 0


class ExportTool(Tool):
    """Export preferences to DPO training format."""

    def __init__(self, parent: Any = None) -> None:
        config = ToolConfig(
            name="export",
            aliases=["e"],
            help_text="Export preferences to DPO training format",
        )
        super().__init__(parent, config)

    def add_args(self, parser) -> None:
        parser.add_argument("--output", "-o", required=True, help="Output JSONL path")
        parser.add_argument("--context", help="Context key filter")
        parser.add_argument("--category", help="Category filter")
        parser.add_argument("--min-margin", type=float, help="Minimum margin threshold")
        parser.add_argument("--since", help="Export records after date (ISO format)")
        parser.add_argument("--until", help="Export records before date (ISO format)")

    def _parse_date(self, date_str: str | None) -> datetime | None:
        """Parse ISO date string to datetime."""
        if date_str is None:
            return None
        return datetime.fromisoformat(date_str)

    def _get_database(self) -> Database:
        """Create Database from config."""
        config = DotDict(**dict(self.app.config)) if self.app.config else DotDict()
        return Database(self.lg, PG(self.lg, config.dbs.main))

    def run(self, **kwargs: Any) -> int:
        from ...training.export import export_preferences_dpo

        db = self._get_database()
        output_path = Path(self.args.output)
        context_key = getattr(self.args, "context", None)
        category = getattr(self.args, "category", None)
        min_margin = getattr(self.args, "min_margin", None)
        since = self._parse_date(getattr(self.args, "since", None))
        until = self._parse_date(getattr(self.args, "until", None))

        self.lg.info(
            "exporting preferences",
            extra={"output": str(output_path), "context": context_key, "category": category},
        )

        result = export_preferences_dpo(
            session_factory=db.session,
            context_key=context_key,
            output_path=output_path,
            category=category,
            since=since,
            until=until,
            min_margin=min_margin,
        )

        self.lg.info(
            "export complete",
            extra={"count": result.count, "path": str(result.path)},
        )
        print(f"Exported {result.count} preference pairs to {result.path}")
        return 0


class DpoTool(ModelResolutionMixin, Tool):
    """Train DPO adapter from preference data."""

    def __init__(self, parent: Any = None) -> None:
        config = ToolConfig(
            name="dpo",
            aliases=["d"],
            help_text="Train DPO adapter from preference data",
        )
        super().__init__(parent, config)

    def add_args(self, parser) -> None:
        parser.add_argument("--data", "-d", help="Input JSONL path")
        parser.add_argument("--output", "-o", help="Output adapter directory")
        parser.add_argument("--model", "-m", help="Model name (from models.yaml)")
        parser.add_argument("--list-models", action="store_true", help="List available models")
        parser.add_argument("--beta", type=float, help="DPO beta parameter (default: 0.1)")
        parser.add_argument(
            "--no-quantize", action="store_true", help="Disable 4-bit quantization"
        )
        parser.add_argument("--epochs", type=int, help="Number of training epochs")
        parser.add_argument("--batch-size", type=int, help="Training batch size")
        parser.add_argument("--lr", type=float, help="Learning rate")
        parser.add_argument("--profile", help="Use named training profile from config")

    def _get_training_config(self) -> DotDict | None:
        """Get training config section."""
        if not self.app.config:
            return None
        return getattr(DotDict(**dict(self.app.config)), "training", None)

    def _load_profile(self, profile_name: str) -> dict:
        """Load named training profile from config."""
        training_cfg = self._get_training_config()
        if training_cfg is None:
            raise ValueError("No training config section found")
        profiles = getattr(training_cfg, "profiles", None)
        if profiles is None:
            raise ValueError("No training.profiles section found")
        profile = getattr(profiles, profile_name, None)
        if profile is None:
            available = list(profiles.keys()) if hasattr(profiles, "keys") else []
            raise ValueError(f"Profile '{profile_name}' not found. Available: {available}")
        return dict(profile)

    def _build_configs(self) -> tuple[float, bool, TrainingConfig]:
        """Build training configuration from args and profile."""
        beta = 0.1
        quantize = True
        training_params: dict[str, Any] = {}

        # Load profile if specified
        profile_name = getattr(self.args, "profile", None)
        if profile_name:
            profile = self._load_profile(profile_name)
            beta = profile.get("beta", beta)
            quantize = profile.get("quantize", quantize)
            if "epochs" in profile:
                training_params["num_epochs"] = profile["epochs"]
            if "batch_size" in profile:
                training_params["batch_size"] = profile["batch_size"]
            if "learning_rate" in profile:
                training_params["learning_rate"] = profile["learning_rate"]

        # CLI args override profile/config
        if getattr(self.args, "beta", None) is not None:
            beta = self.args.beta
        if getattr(self.args, "no_quantize", False):
            quantize = False
        if getattr(self.args, "epochs", None) is not None:
            training_params["num_epochs"] = self.args.epochs
        if getattr(self.args, "batch_size", None) is not None:
            training_params["batch_size"] = self.args.batch_size
        if getattr(self.args, "lr", None) is not None:
            training_params["learning_rate"] = self.args.lr

        training_config = TrainingConfig(**training_params) if training_params else TrainingConfig()
        return beta, quantize, training_config

    def run(self, **kwargs: Any) -> int:
        # Handle --list-models
        if getattr(self.args, "list_models", False):
            return self._list_models()

        # Validate required args
        if not getattr(self.args, "data", None):
            self.lg.error("--data is required")
            return 1
        if not getattr(self.args, "output", None):
            self.lg.error("--output is required")
            return 1

        from ...training.dpo import train_dpo

        data_path = Path(self.args.data)
        output_dir = Path(self.args.output)

        if not data_path.exists():
            self.lg.error("data file not found", extra={"path": str(data_path)})
            return 1

        # Resolve model
        model_name = getattr(self.args, "model", None)
        model_path = self._resolve_model(model_name)
        if model_path is None:
            return 1

        beta, quantize, training_config = self._build_configs()

        self.lg.info(
            "starting DPO training",
            extra={
                "data": str(data_path),
                "output": str(output_dir),
                "model": str(model_path),
                "beta": beta,
                "quantize": quantize,
                "epochs": training_config.num_epochs,
            },
        )

        result = train_dpo(
            lg=self.lg,
            data_path=data_path,
            output_dir=output_dir,
            base_model=str(model_path),
            lora_config=LoraConfig(),
            training_config=training_config,
            beta=beta,
            quantize=quantize,
        )

        self._print_result(result)
        return 0

    def _print_result(self, result: TrainingResult) -> None:
        """Print training result summary."""
        print("\nTraining complete!")
        print(f"  Adapter: {result.adapter_path}")
        print(f"  Base model: {result.base_model}")
        print(f"  Duration: {result.duration_seconds:.1f}s")
        print(f"  Samples trained: {result.samples_trained}")
        if result.metrics:
            print("  Metrics:")
            for key, value in result.metrics.items():
                if isinstance(value, float):
                    print(f"    {key}: {value:.4f}")
                else:
                    print(f"    {key}: {value}")


class RegisterTool(Tool):
    """Copy adapter weights to llm-infer adapter directory."""

    def __init__(self, parent: Any = None) -> None:
        config = ToolConfig(
            name="register",
            aliases=["r"],
            help_text="Copy adapter weights to llm-infer adapter directory",
        )
        super().__init__(parent, config)

    def add_args(self, parser) -> None:
        parser.add_argument("--adapter", "-a", required=True, help="Adapter directory path")
        parser.add_argument("--id", required=True, help="Adapter ID for llm-infer")
        parser.add_argument("--description", help="Optional description")
        parser.add_argument("--disabled", action="store_true", help="Register but don't enable")
        parser.add_argument("--overwrite", action="store_true", help="Overwrite existing adapter")

    def _get_base_path(self) -> str:
        """Get adapter base path from config."""
        config = DotDict(**dict(self.app.config)) if self.app.config else DotDict()
        adapters_cfg = getattr(config, "adapters", None)
        lora_cfg = getattr(adapters_cfg, "lora", None) if adapters_cfg else None
        base_path = lora_cfg.base_path if lora_cfg and hasattr(lora_cfg, "base_path") else None

        if base_path is None:
            raise ValueError("adapters.lora.base_path not configured")

        return base_path

    def _build_training_result(self, adapter_path: Path) -> TrainingResult:
        """Build a minimal TrainingResult for registration."""
        from ...core.utils import utc_now

        return TrainingResult(
            adapter_path=adapter_path,
            base_model="unknown",
            method="dpo",
            metrics={},
            config={},
            started_at=utc_now(),
            completed_at=utc_now(),
            samples_trained=0,
        )

    def run(self, **kwargs: Any) -> int:
        from ...training.registry import AdapterRegistry

        adapter_path = Path(self.args.adapter)
        adapter_id = self.args.id
        description = getattr(self.args, "description", None)
        enabled = not getattr(self.args, "disabled", False)
        overwrite = getattr(self.args, "overwrite", False)

        if not adapter_path.exists():
            self.lg.error("adapter path not found", extra={"path": str(adapter_path)})
            return 1

        base_path = self._get_base_path()
        registry = AdapterRegistry(self.lg, base_path)

        training_result = self._build_training_result(adapter_path)

        self.lg.info(
            "registering adapter",
            extra={"adapter_id": adapter_id, "path": str(adapter_path), "enabled": enabled},
        )

        try:
            info = registry.register(
                training_result=training_result,
                adapter_id=adapter_id,
                description=description,
                enabled=enabled,
                overwrite=overwrite,
            )
        except ValueError as e:
            self.lg.error("registration failed", extra={"error": str(e)})
            return 1

        print(f"Registered adapter '{info.adapter_id}' to {info.path}")
        if info.enabled:
            print("  Status: enabled")
        else:
            print("  Status: disabled")
        return 0


class PipelineTool(ModelResolutionMixin, Tool):
    """Run full DPO workflow: export -> train -> register."""

    def __init__(self, parent: Any = None) -> None:
        config = ToolConfig(
            name="pipeline",
            aliases=["p"],
            help_text="Run full DPO workflow: export -> train -> register",
        )
        super().__init__(parent, config)

    def add_args(self, parser) -> None:
        # Export args
        parser.add_argument("--context", help="Context key for export")
        parser.add_argument("--category", help="Category filter for export")
        parser.add_argument("--min-margin", type=float, help="Minimum margin threshold")

        # Training args
        parser.add_argument("--id", help="Adapter ID for registration")
        parser.add_argument("--output-dir", help="Base output directory")
        parser.add_argument("--profile", help="Training profile name")
        parser.add_argument("--model", "-m", help="Model name (from models.yaml)")
        parser.add_argument("--list-models", action="store_true", help="List available models")
        parser.add_argument("--beta", type=float, help="DPO beta parameter")
        parser.add_argument("--no-quantize", action="store_true", help="Disable quantization")
        parser.add_argument("--epochs", type=int, help="Number of epochs")

        # Register args
        parser.add_argument("--description", help="Adapter description")
        parser.add_argument("--overwrite", action="store_true", help="Overwrite existing adapter")
        parser.add_argument("--skip-register", action="store_true", help="Skip registration step")

    def _get_output_dir(self) -> Path:
        """Get output directory from args or config."""
        if getattr(self.args, "output_dir", None):
            return Path(self.args.output_dir)

        config = DotDict(**dict(self.app.config)) if self.app.config else DotDict()
        training_cfg = getattr(config, "training", None)
        if training_cfg and hasattr(training_cfg, "output_dir"):
            return Path(training_cfg.output_dir)

        return Path("./training-output")

    def _run_export(self, output_path: Path) -> int:
        """Run export step."""
        from ...training.export import export_preferences_dpo

        config = DotDict(**dict(self.app.config)) if self.app.config else DotDict()
        db = Database(self.lg, PG(self.lg, config.dbs.main))

        context_key = self.args.context
        category = getattr(self.args, "category", None)
        min_margin = getattr(self.args, "min_margin", None)

        self.lg.info("step 1/3: exporting preferences", extra={"context": context_key})

        result = export_preferences_dpo(
            session_factory=db.session,
            context_key=context_key,
            output_path=output_path,
            category=category,
            min_margin=min_margin,
        )

        if result.count == 0:
            self.lg.error("no preferences found to export")
            return 1

        print(f"[1/3] Exported {result.count} preference pairs")
        return 0

    def _run_train(self, data_path: Path, adapter_path: Path, model_path: Path) -> TrainingResult | None:
        """Run training step."""
        from ...training.dpo import train_dpo

        beta = 0.1
        quantize = True
        training_params: dict[str, Any] = {}

        config = DotDict(**dict(self.app.config)) if self.app.config else DotDict()
        training_cfg = getattr(config, "training", None)

        # Load profile if specified
        if training_cfg:
            profile_name = getattr(self.args, "profile", None)
            if profile_name:
                profiles = getattr(training_cfg, "profiles", None)
                if profiles:
                    profile = getattr(profiles, profile_name, None)
                    if profile:
                        profile_dict = dict(profile)
                        beta = profile_dict.get("beta", beta)
                        quantize = profile_dict.get("quantize", quantize)
                        if "epochs" in profile_dict:
                            training_params["num_epochs"] = profile_dict["epochs"]
                        if "batch_size" in profile_dict:
                            training_params["batch_size"] = profile_dict["batch_size"]
                        if "learning_rate" in profile_dict:
                            training_params["learning_rate"] = profile_dict["learning_rate"]

        # CLI overrides
        if getattr(self.args, "beta", None) is not None:
            beta = self.args.beta
        if getattr(self.args, "no_quantize", False):
            quantize = False
        if getattr(self.args, "epochs", None) is not None:
            training_params["num_epochs"] = self.args.epochs

        training_config = TrainingConfig(**training_params) if training_params else TrainingConfig()

        self.lg.info(
            "step 2/3: training DPO adapter",
            extra={"model": str(model_path), "epochs": training_config.num_epochs},
        )

        try:
            result = train_dpo(
                lg=self.lg,
                data_path=data_path,
                output_dir=adapter_path,
                base_model=str(model_path),
                lora_config=LoraConfig(),
                training_config=training_config,
                beta=beta,
                quantize=quantize,
            )
            print(f"[2/3] Training complete ({result.duration_seconds:.1f}s)")
            return result
        except Exception as e:
            self.lg.error("training failed", extra={"exception": e})
            return None

    def _run_register(self, training_result: TrainingResult) -> int:
        """Run registration step."""
        from ...training.registry import AdapterRegistry

        config = DotDict(**dict(self.app.config)) if self.app.config else DotDict()
        adapters_cfg = getattr(config, "adapters", None)
        lora_cfg = getattr(adapters_cfg, "lora", None) if adapters_cfg else None
        base_path = lora_cfg.base_path if lora_cfg and hasattr(lora_cfg, "base_path") else None

        if base_path is None:
            self.lg.error("adapters.lora.base_path not configured")
            return 1

        registry = AdapterRegistry(self.lg, base_path)

        adapter_id = self.args.id
        description = getattr(self.args, "description", None)
        overwrite = getattr(self.args, "overwrite", False)

        self.lg.info("step 3/3: registering adapter", extra={"adapter_id": adapter_id})

        try:
            info = registry.register(
                training_result=training_result,
                adapter_id=adapter_id,
                description=description,
                enabled=True,
                overwrite=overwrite,
            )
            print(f"[3/3] Registered adapter '{info.adapter_id}'")
            return 0
        except ValueError as e:
            self.lg.error("registration failed", extra={"error": str(e)})
            return 1

    def run(self, **kwargs: Any) -> int:
        # Handle --list-models
        if getattr(self.args, "list_models", False):
            return self._list_models()

        # Validate required args
        if not getattr(self.args, "context", None):
            self.lg.error("--context is required")
            return 1
        if not getattr(self.args, "id", None):
            self.lg.error("--id is required")
            return 1

        # Resolve model early
        model_name = getattr(self.args, "model", None)
        model_path = self._resolve_model(model_name)
        if model_path is None:
            return 1

        adapter_id = self.args.id
        output_dir = self._get_output_dir()

        # Set up paths
        work_dir = output_dir / adapter_id
        data_path = work_dir / "data.jsonl"
        adapter_path = work_dir / "adapter"

        work_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nDPO Pipeline: {self.args.context} -> {adapter_id}")
        print(f"Model: {model_path.name}")
        print(f"Output directory: {work_dir}\n")

        # Step 1: Export
        if self._run_export(data_path) != 0:
            print("\nPipeline failed at step 1 (export)")
            return 1

        # Step 2: Train
        training_result = self._run_train(data_path, adapter_path, model_path)
        if training_result is None:
            print("\nPipeline failed at step 2 (training)")
            return 1

        # Step 3: Register (optional)
        if getattr(self.args, "skip_register", False):
            print(f"\nTraining complete! Adapter at: {training_result.adapter_path}")
            return 0

        if self._run_register(training_result) != 0:
            print("\nPipeline failed at step 3 (registration)")
            return 1

        print(f"\nPipeline complete! Adapter '{adapter_id}' is ready.")
        return 0


class TrainTool(Tool):
    """Training commands for DPO workflow."""

    def __init__(self, parent: Any = None) -> None:
        config = ToolConfig(
            name="train",
            aliases=["t"],
            help_text="Training commands (export, dpo, register, pipeline)",
        )
        super().__init__(parent, config)
        self.add_tool(ExportTool(self))
        self.add_tool(DpoTool(self))
        self.add_tool(RegisterTool(self))
        self.add_tool(PipelineTool(self))

    def run(self, **kwargs: Any) -> int:
        """Delegate to subtool."""
        result: int = self.group.run(**kwargs)
        return result
