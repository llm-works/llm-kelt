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
        models = getattr(config, "models", None)
        return models if isinstance(models, DotDict) else DotDict()

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
        try:
            return datetime.fromisoformat(date_str)
        except ValueError as e:
            raise ValueError(
                f"Invalid date format '{date_str}': expected ISO format (e.g., 2024-01-15)"
            ) from e

    def _get_database(self) -> Database:
        """Create Database from config."""
        config = DotDict(**dict(self.app.config)) if self.app.config else DotDict()
        dbs = getattr(config, "dbs", None)
        if dbs is None or not hasattr(dbs, "main"):
            raise ValueError("Database not configured: missing dbs.main in config")
        return Database(self.lg, PG(self.lg, dbs.main))

    def _get_export_filters(self) -> dict[str, Any]:
        """Extract export filters from args."""
        return {
            "context_key": getattr(self.args, "context", None),
            "category": getattr(self.args, "category", None),
            "min_margin": getattr(self.args, "min_margin", None),
            "since": self._parse_date(getattr(self.args, "since", None)),
            "until": self._parse_date(getattr(self.args, "until", None)),
        }

    def run(self, **kwargs: Any) -> int:
        from ...training.export import export_preferences_dpo

        db = self._get_database()
        output_path = Path(self.args.output)
        filters = self._get_export_filters()

        self.lg.info("exporting preferences", extra={"output": str(output_path), **filters})

        result = export_preferences_dpo(
            session_factory=db.session, output_path=output_path, **filters
        )

        self.lg.info("export complete", extra={"count": result.count, "path": str(result.path)})
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
        parser.add_argument("--no-quantize", action="store_true", help="Disable 4-bit quantization")
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

    def _apply_profile(self, profile: dict) -> tuple[float, bool, dict[str, Any]]:
        """Apply profile settings, returning (beta, quantize, params)."""
        params: dict[str, Any] = {}
        for key, param in [
            ("epochs", "num_epochs"),
            ("batch_size", "batch_size"),
            ("learning_rate", "learning_rate"),
        ]:
            if key in profile:
                params[param] = profile[key]
        return profile.get("beta", 0.1), profile.get("quantize", True), params

    def _apply_cli_overrides(
        self, beta: float, quantize: bool, params: dict
    ) -> tuple[float, bool, dict]:
        """Apply CLI arg overrides to profile settings."""
        if getattr(self.args, "beta", None) is not None:
            beta = self.args.beta
        if getattr(self.args, "no_quantize", False):
            quantize = False
        for arg, param in [
            ("epochs", "num_epochs"),
            ("batch_size", "batch_size"),
            ("lr", "learning_rate"),
        ]:
            if getattr(self.args, arg, None) is not None:
                params[param] = getattr(self.args, arg)
        return beta, quantize, params

    def _build_configs(self) -> tuple[float, bool, TrainingConfig]:
        """Build training configuration from args and profile."""
        params: dict[str, Any] = {}
        beta, quantize = 0.1, True

        profile_name = getattr(self.args, "profile", None)
        if profile_name:
            beta, quantize, params = self._apply_profile(self._load_profile(profile_name))

        beta, quantize, params = self._apply_cli_overrides(beta, quantize, params)
        return beta, quantize, TrainingConfig(**params) if params else TrainingConfig()

    def _validate_args(self) -> tuple[Path, Path] | None:
        """Validate args and return (data_path, output_dir) or None on error."""
        if not getattr(self.args, "data", None):
            self.lg.error("--data is required")
            return None
        if not getattr(self.args, "output", None):
            self.lg.error("--output is required")
            return None

        data_path = Path(self.args.data)
        if not data_path.exists():
            self.lg.error("data file not found", extra={"path": str(data_path)})
            return None

        return data_path, Path(self.args.output)

    def run(self, **kwargs: Any) -> int:
        if getattr(self.args, "list_models", False):
            return self._list_models()

        paths = self._validate_args()
        if paths is None:
            return 1
        data_path, output_dir = paths

        model_path = self._resolve_model(getattr(self.args, "model", None))
        if model_path is None:
            return 1

        return self._execute_training(data_path, output_dir, model_path)

    def _execute_training(self, data_path: Path, output_dir: Path, model_path: Path) -> int:
        """Execute the DPO training."""
        from ...training.dpo import train_dpo

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
        if not lora_cfg or not hasattr(lora_cfg, "base_path"):
            raise ValueError("adapters.lora.base_path not configured")

        return str(lora_cfg.base_path)

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

    def _do_register(self, adapter_path: Path, adapter_id: str, enabled: bool) -> int:
        """Execute registration and print result."""
        from ...training.lora import AdapterRegistry

        registry = AdapterRegistry(self.lg, self._get_base_path())
        training_result = self._build_training_result(adapter_path)

        self.lg.info("registering adapter", extra={"adapter_id": adapter_id, "enabled": enabled})

        try:
            info = registry.register(
                training_result=training_result,
                adapter_id=adapter_id,
                description=getattr(self.args, "description", None),
                enabled=enabled,
                overwrite=getattr(self.args, "overwrite", False),
            )
        except ValueError as e:
            self.lg.error("registration failed", extra={"error": str(e)})
            return 1

        status = "enabled" if info.enabled else "disabled"
        print(f"Registered adapter '{info.adapter_id}' to {info.path}\n  Status: {status}")
        return 0

    def run(self, **kwargs: Any) -> int:
        adapter_path = Path(self.args.adapter)
        if not adapter_path.exists():
            self.lg.error("adapter path not found", extra={"path": str(adapter_path)})
            return 1

        enabled = not getattr(self.args, "disabled", False)
        return self._do_register(adapter_path, self.args.id, enabled)


class PipelineTool(ModelResolutionMixin, Tool):
    """Run full DPO workflow: export -> train -> register."""

    def __init__(self, parent: Any = None) -> None:
        config = ToolConfig(
            name="pipeline",
            aliases=["p"],
            help_text="Run full DPO workflow: export -> train -> register",
        )
        super().__init__(parent, config)
        self._db: Database | None = None
        self._dpo_client = None
        self._run_id: int | None = None

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
        parser.add_argument("--batch-size", type=int, help="Training batch size")
        parser.add_argument("--lr", type=float, help="Learning rate")

        # Register args
        parser.add_argument("--description", help="Adapter description")
        parser.add_argument("--overwrite", action="store_true", help="Overwrite existing adapter")
        parser.add_argument("--skip-register", action="store_true", help="Skip registration step")

        # Run tracking
        parser.add_argument("--run-id", type=int, help="Use existing DPO run ID")

    def _get_output_dir(self) -> Path:
        """Get output directory from args or config."""
        if getattr(self.args, "output_dir", None):
            return Path(self.args.output_dir)

        config = DotDict(**dict(self.app.config)) if self.app.config else DotDict()
        training_cfg = getattr(config, "training", None)
        if training_cfg and hasattr(training_cfg, "output_dir"):
            return Path(training_cfg.output_dir)

        return Path("./training-output")

    def _get_database(self) -> Database:
        """Get or create cached Database instance."""
        if self._db is None:
            config = DotDict(**dict(self.app.config)) if self.app.config else DotDict()
            dbs = getattr(config, "dbs", None)
            if dbs is None or not hasattr(dbs, "main"):
                raise ValueError("Database not configured: missing dbs.main in config")
            self._db = Database(self.lg, PG(self.lg, dbs.main))
        return self._db

    def _get_dpo_client(self):
        """Get or create DpoClient for run tracking."""
        if self._dpo_client is None:
            from ...training.dpo import DpoClient

            self._dpo_client = DpoClient(
                lg=self.lg,
                session_factory=self._get_database().session,
                context_key=self.args.context,
            )
        return self._dpo_client

    def _get_or_create_run(self) -> int:
        """Get existing run or create new one."""
        client = self._get_dpo_client()

        # Use existing run if specified
        if run_id := getattr(self.args, "run_id", None):
            run = client.get(run_id)
            if run is None:
                raise ValueError(f"DPO run {run_id} not found")
            if run.status != "pending":
                raise ValueError(f"DPO run {run_id} is {run.status}, expected pending")
            return int(run_id)

        # Look for existing pending run with same adapter name
        pending = client.list(status="pending")
        for run in pending:
            if run.adapter_name == self.args.id:
                self.lg.info("using existing pending run", extra={"run_id": run.id})
                return int(run.id)

        # Create new run
        run = client.create(adapter_name=self.args.id)
        self.lg.info("created new DPO run", extra={"run_id": run.id})
        return int(run.id)

    def _run_export(self, output_path: Path) -> int:
        """Run export step."""
        from ...training.export import export_preferences_dpo

        context_key = self.args.context
        category = getattr(self.args, "category", None)
        min_margin = getattr(self.args, "min_margin", None)

        self.lg.info("step 1/3: exporting preferences", extra={"context": context_key})

        result = export_preferences_dpo(
            session_factory=self._get_database().session,
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

    def _load_profile(self, profile_name: str) -> tuple[float, bool, dict[str, Any]]:
        """Load and validate a training profile."""
        config = DotDict(**dict(self.app.config)) if self.app.config else DotDict()
        training_cfg = getattr(config, "training", None)
        if training_cfg is None:
            raise ValueError("No training config section found")
        profiles = getattr(training_cfg, "profiles", None)
        if profiles is None:
            raise ValueError("No training.profiles section found")
        profile = getattr(profiles, profile_name, None)
        if profile is None:
            available = list(profiles.keys()) if hasattr(profiles, "keys") else []
            raise ValueError(f"Profile '{profile_name}' not found. Available: {available}")

        profile_dict = dict(profile)
        params: dict[str, Any] = {}
        for key, param in [
            ("epochs", "num_epochs"),
            ("batch_size", "batch_size"),
            ("learning_rate", "learning_rate"),
        ]:
            if key in profile_dict:
                params[param] = profile_dict[key]
        return profile_dict.get("beta", 0.1), profile_dict.get("quantize", True), params

    def _apply_cli_overrides(
        self, beta: float, quantize: bool, params: dict
    ) -> tuple[float, bool, dict]:
        """Apply CLI argument overrides to profile settings."""
        if getattr(self.args, "beta", None) is not None:
            beta = self.args.beta
        if getattr(self.args, "no_quantize", False):
            quantize = False
        for arg, param in [
            ("epochs", "num_epochs"),
            ("batch_size", "batch_size"),
            ("lr", "learning_rate"),
        ]:
            if getattr(self.args, arg, None) is not None:
                params[param] = getattr(self.args, arg)
        return beta, quantize, params

    def _build_training_params(self) -> tuple[float, bool, TrainingConfig]:
        """Build training parameters from profile and CLI args."""
        params: dict[str, Any] = {}
        beta, quantize = 0.1, True

        if profile_name := getattr(self.args, "profile", None):
            beta, quantize, params = self._load_profile(profile_name)

        beta, quantize, params = self._apply_cli_overrides(beta, quantize, params)
        return beta, quantize, TrainingConfig(**params) if params else TrainingConfig()

    def _run_train(
        self, data_path: Path, adapter_path: Path, model_path: Path
    ) -> TrainingResult | None:
        """Run training step."""
        from ...training.dpo import train_dpo

        beta, quantize, training_config = self._build_training_params()

        self.lg.info(
            "step 2/3: training DPO adapter",
            extra={"model": str(model_path), "epochs": training_config.num_epochs},
        )

        try:
            result: TrainingResult = train_dpo(
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

    def _get_adapter_base_path(self) -> str | None:
        """Get adapter base path from config."""
        config = DotDict(**dict(self.app.config)) if self.app.config else DotDict()
        adapters_cfg = getattr(config, "adapters", None)
        lora_cfg = getattr(adapters_cfg, "lora", None) if adapters_cfg else None
        return lora_cfg.base_path if lora_cfg and hasattr(lora_cfg, "base_path") else None

    def _do_register(self, registry, training_result: TrainingResult, overwrite: bool):
        """Perform the actual adapter registration."""
        return registry.register(
            training_result=training_result,
            adapter_id=self.args.id,
            description=getattr(self.args, "description", None),
            enabled=True,
            overwrite=overwrite,
        )

    def _run_register(self, training_result: TrainingResult) -> int:
        """Run registration step."""
        from ...training.lora import AdapterRegistry

        base_path = self._get_adapter_base_path()
        if base_path is None:
            self.lg.error("adapters.lora.base_path not configured")
            return 1

        registry = AdapterRegistry(self.lg, base_path)
        self.lg.info("step 3/3: registering adapter", extra={"adapter_id": self.args.id})
        overwrite = getattr(self.args, "overwrite", False)

        try:
            info = self._do_register(registry, training_result, overwrite)
            print(f"[3/3] Registered adapter '{info.adapter_id}'")
            return 0
        except ValueError as e:
            error_msg = str(e)
            if "already exists" in error_msg and not overwrite:
                response = input(f"\nAdapter '{self.args.id}' already exists. Overwrite? [y/N] ")
                if response.lower() in ("y", "yes"):
                    info = self._do_register(registry, training_result, overwrite=True)
                    print(f"[3/3] Registered adapter '{info.adapter_id}'")
                    return 0
                print("Registration skipped.")
                return 1
            self.lg.error("registration failed", extra={"error": error_msg})
            return 1

    def _validate_args(self) -> bool:
        """Validate required arguments."""
        if not getattr(self.args, "context", None):
            self.lg.error("--context is required")
            return False
        if not getattr(self.args, "id", None):
            self.lg.error("--id is required")
            return False
        return True

    def _setup_paths(self, adapter_id: str) -> tuple[Path, Path, Path]:
        """Set up working directory and paths."""
        output_dir = self._get_output_dir()
        work_dir = output_dir / adapter_id
        work_dir.mkdir(parents=True, exist_ok=True)
        return work_dir, work_dir / "data.jsonl", work_dir / "adapter"

    def run(self, **kwargs: Any) -> int:
        if getattr(self.args, "list_models", False):
            return self._list_models()

        if not self._validate_args():
            return 1

        model_path = self._resolve_model(getattr(self.args, "model", None))
        if model_path is None:
            return 1

        # Get or create DPO run for tracking
        try:
            self._run_id = self._get_or_create_run()
        except ValueError as e:
            self.lg.error(str(e))
            return 1

        adapter_id = self.args.id
        work_dir, data_path, adapter_path = self._setup_paths(adapter_id)

        print(f"\nDPO Pipeline: {self.args.context} -> {adapter_id}")
        print(f"Model: {model_path.name}")
        print(f"Output directory: {work_dir}")
        print(f"Run ID: {self._run_id}\n")

        return self._execute_pipeline(data_path, adapter_path, model_path)

    def _execute_pipeline(self, data_path: Path, adapter_path: Path, model_path: Path) -> int:
        """Execute the three pipeline steps with run tracking."""
        client = self._get_dpo_client()

        if self._run_export(data_path) != 0:
            client.fail(self._run_id, "Export failed: no preferences found")
            return self._pipeline_fail("step 1 (export)")

        client.start(self._run_id)
        training_result = self._run_train(data_path, adapter_path, model_path)
        if training_result is None:
            client.fail(self._run_id, "Training failed")
            return self._pipeline_fail("step 2 (training)")

        return self._finish_pipeline(client, training_result)

    def _pipeline_fail(self, step: str) -> int:
        """Print pipeline failure message and return error code."""
        print(f"\nPipeline failed at {step}")
        return 1

    def _finish_pipeline(self, client, training_result: TrainingResult) -> int:
        """Handle registration and completion after successful training."""
        skip_register = getattr(self.args, "skip_register", False)
        reg_failed = False if skip_register else self._run_register(training_result) != 0

        # Always record metrics if training succeeded
        client.complete(self._run_id, metrics=self._extract_metrics(training_result))

        if skip_register:
            print(f"\nTraining complete! Adapter at: {training_result.adapter_path}")
        elif reg_failed:
            print("\nTraining complete, but registration failed (see above)")
            return 1
        else:
            print(f"\nPipeline complete! Adapter '{self.args.id}' is ready.")
        return 0

    def _extract_metrics(self, result: TrainingResult) -> dict:
        """Extract metrics from training result for DB storage."""
        metrics = dict(result.metrics) if result.metrics else {}
        metrics["duration_seconds"] = result.duration_seconds
        metrics["samples_trained"] = result.samples_trained
        return metrics


class ResetTool(Tool):
    """Reset DPO training data for a context."""

    # Display name for NULL context_key (used in list output)
    _NULL_CONTEXT_DISPLAY = "(no context)"

    def __init__(self, parent: Any = None) -> None:
        config = ToolConfig(
            name="reset",
            help_text="Reset DPO training runs for a context",
        )
        super().__init__(parent, config)

    def add_args(self, parser) -> None:
        parser.add_argument(
            "--context",
            default=None,
            help="Context key to reset (omit to list contexts)",
        )
        parser.add_argument("--all", action="store_true", help="Reset ALL contexts")
        parser.add_argument("--confirm", action="store_true", help="Skip confirmation prompt")

    def _get_database(self) -> Database:
        """Get Database instance from config."""
        config = DotDict(**dict(self.app.config)) if self.app.config else DotDict()
        dbs = getattr(config, "dbs", None)
        if dbs is None or not hasattr(dbs, "main"):
            raise ValueError("Database not configured: missing dbs.main in config")
        return Database(self.lg, PG(self.lg, dbs.main))

    def _normalize_context_key(self, context_key: str) -> str | None:
        """Convert display name back to database value (handles NULL context)."""
        return None if context_key == self._NULL_CONTEXT_DISPLAY else context_key

    def _context_filter(self, column, context_key: str | None):
        """Build SQLAlchemy filter for context_key (handles NULL correctly)."""
        return column.is_(None) if context_key is None else column == context_key

    def _list_contexts(self, session) -> list[tuple[str, int, int]]:
        """List all contexts with run and pair counts (excludes deleted runs)."""
        from sqlalchemy import func, select

        from ...training.dpo import DpoRun, DpoRunPair

        # Get distinct contexts with run counts (excluding deleted)
        stmt = (
            select(DpoRun.context_key, func.count(DpoRun.id).label("run_count"))
            .where(DpoRun.status != "deleted")
            .group_by(DpoRun.context_key)
            .order_by(DpoRun.context_key)
        )
        results = []
        for row in session.execute(stmt):
            display_key = row.context_key or self._NULL_CONTEXT_DISPLAY
            run_count = row.run_count

            # Count pairs for this context (use _context_filter for NULL handling)
            pair_stmt = (
                select(func.count())
                .select_from(DpoRunPair)
                .join(DpoRun, DpoRunPair.run_id == DpoRun.id)
                .where(
                    self._context_filter(DpoRun.context_key, row.context_key),
                    DpoRun.status != "deleted",
                )
            )
            pair_count = session.scalar(pair_stmt) or 0
            results.append((display_key, run_count, pair_count))

        return results

    def _count_data(self, session, context_key: str | None) -> tuple[int, int]:
        """Count runs and pairs for the context (excludes deleted runs)."""
        from sqlalchemy import func, select

        from ...training.dpo import DpoRun, DpoRunPair

        context_filter = self._context_filter(DpoRun.context_key, context_key)

        # Count runs (excluding deleted)
        run_count_stmt = (
            select(func.count())
            .select_from(DpoRun)
            .where(context_filter, DpoRun.status != "deleted")
        )
        run_count = session.scalar(run_count_stmt) or 0

        # Count pairs linked to those runs
        pair_count_stmt = (
            select(func.count())
            .select_from(DpoRunPair)
            .join(DpoRun, DpoRunPair.run_id == DpoRun.id)
            .where(context_filter)
        )
        pair_count = session.scalar(pair_count_stmt) or 0

        return run_count, pair_count

    def _delete_data(self, session, context_key: str | None) -> tuple[int, int]:
        """Soft-delete runs for the context. Returns (runs_deleted, pairs_freed)."""
        from sqlalchemy import delete, select, update

        from ...training.dpo import DpoRun, DpoRunPair

        context_filter = self._context_filter(DpoRun.context_key, context_key)

        # Get run IDs for this context (excluding already deleted)
        run_ids_stmt = select(DpoRun.id).where(context_filter, DpoRun.status != "deleted")
        run_ids = list(session.scalars(run_ids_stmt).all())

        if not run_ids:
            return 0, 0

        # Delete pairs to free them for reuse (hard delete - they're just links)
        pairs_stmt = delete(DpoRunPair).where(DpoRunPair.run_id.in_(run_ids))
        pairs_result = session.execute(pairs_stmt)
        pairs_deleted = pairs_result.rowcount

        # Soft-delete runs (set status to 'deleted')
        runs_stmt = update(DpoRun).where(DpoRun.id.in_(run_ids)).values(status="deleted")
        runs_result = session.execute(runs_stmt)
        runs_deleted = runs_result.rowcount

        return runs_deleted, pairs_deleted

    def _count_all_data(self, session) -> tuple[int, int]:
        """Count all runs and pairs across all contexts (excludes deleted)."""
        from sqlalchemy import func, select

        from ...training.dpo import DpoRun, DpoRunPair

        run_count = (
            session.scalar(
                select(func.count()).select_from(DpoRun).where(DpoRun.status != "deleted")
            )
            or 0
        )
        pair_count = session.scalar(select(func.count()).select_from(DpoRunPair)) or 0
        return run_count, pair_count

    def _delete_all_data(self, session) -> tuple[int, int]:
        """Soft-delete all runs. Returns (runs_deleted, pairs_freed)."""
        from sqlalchemy import delete, select, update

        from ...training.dpo import DpoRun, DpoRunPair

        # Get IDs of non-deleted runs
        run_ids = list(session.scalars(select(DpoRun.id).where(DpoRun.status != "deleted")).all())

        if not run_ids:
            return 0, 0

        # Delete pairs to free them (hard delete - they're just links)
        pairs_result = session.execute(delete(DpoRunPair).where(DpoRunPair.run_id.in_(run_ids)))
        pairs_deleted = pairs_result.rowcount

        # Soft-delete runs
        runs_result = session.execute(
            update(DpoRun).where(DpoRun.id.in_(run_ids)).values(status="deleted")
        )
        runs_deleted = runs_result.rowcount

        return runs_deleted, pairs_deleted

    def run(self, **kwargs: Any) -> int:
        context_key = getattr(self.args, "context", None)
        reset_all = getattr(self.args, "all", False)
        confirm = getattr(self.args, "confirm", False)

        db = self._get_database()

        with db.session() as session:
            # Handle --all: reset everything
            if reset_all:
                return self._run_reset_all(session, confirm)

            # List contexts if none specified
            if not context_key:
                return self._run_list_contexts(session)

            # Reset specific context
            return self._run_reset_context(session, context_key, confirm)

    def _run_list_contexts(self, session) -> int:
        """List available contexts."""
        contexts = self._list_contexts(session)
        if not contexts:
            print("No training runs found.")
            return 0

        print("\nAvailable contexts:\n")
        print(f"  {'Context':<30} {'Runs':>8} {'Pairs':>10}")
        print(f"  {'-' * 30} {'-' * 8} {'-' * 10}")
        for ctx, runs, pairs in contexts:
            print(f"  {ctx:<30} {runs:>8} {pairs:>10}")
        print("\nUse --context <name> to reset a specific context.")
        return 0

    def _run_reset_context(self, session, context_key: str, confirm: bool) -> int:
        """Reset a specific context."""
        # Convert display name to database value (handles "(no context)" -> None)
        db_context_key = self._normalize_context_key(context_key)
        run_count, pair_count = self._count_data(session, db_context_key)

        if run_count == 0:
            print(f"No training data found for context '{context_key}'")
            return 0

        print(f"\nThis will delete for context '{context_key}':")
        print(f"  Training runs:    {run_count}")
        print(f"  Run-pair links:   {pair_count}")
        print()

        if not confirm:
            response = input("Proceed? [y/N] ").strip().lower()
            if response != "y":
                print("Aborted.")
                return 1

        runs_deleted, pairs_deleted = self._delete_data(session, db_context_key)
        session.commit()

        print(f"Deleted {runs_deleted} runs, {pairs_deleted} run-pair links")
        return 0

    def _run_reset_all(self, session, confirm: bool) -> int:
        """Reset all contexts."""
        run_count, pair_count = self._count_all_data(session)

        if run_count == 0:
            print("No training data found.")
            return 0

        print("\nThis will delete ALL training data:")
        print(f"  Training runs:    {run_count}")
        print(f"  Run-pair links:   {pair_count}")
        print()

        if not confirm:
            response = input("Are you sure? [y/N] ").strip().lower()
            if response != "y":
                print("Aborted.")
                return 1

        runs_deleted, pairs_deleted = self._delete_all_data(session)
        session.commit()

        print(f"Deleted {runs_deleted} runs, {pairs_deleted} run-pair links")
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
        self.add_tool(ResetTool(self))

    def run(self, **kwargs: Any) -> int:
        """Delegate to subtool."""
        result: int = self.group.run(**kwargs)
        return result
