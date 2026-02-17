"""Training tools - CLI for DPO training workflow."""

from datetime import datetime
from pathlib import Path
from typing import Any

from appinfra.app.tools import Tool, ToolConfig
from appinfra.db.pg import PG
from appinfra.dot_dict import DotDict
from llm_infer import compute_adapter_metadata
from llm_infer.models import ModelResolver
from sqlalchemy import delete, func, literal, select

from ...core.database import Database
from ...core.utils import utc_now
from ...training.config import RunConfig, RunResult
from ...training.dpo import (
    PendingPair,
    Run,
    TrainedPair,
    _not_deleted_filter,
    export_preferences,
    export_run_pairs,
)
from ...training.lora import AdapterRegistry
from ...training.lora import Config as LoraConfig
from ...training.sft import PendingExample, TrainedExample, export_run_examples


def _print_adapter_metadata(adapter_path: Path, label: str = "Adapter") -> None:
    """Print adapter metadata (mtime, md5) for verification."""
    meta = compute_adapter_metadata(adapter_path)
    if meta.md5 == "unknown":
        print(f"  {label} metadata: (weights file not found)")
    else:
        print(f"  {label} metadata: mtime={meta.mtime}, md5={meta.md5}")


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
        db = self._get_database()
        output_path = Path(self.args.output)
        filters = self._get_export_filters()

        self.lg.info("exporting preferences", extra={"output": str(output_path), **filters})

        result = export_preferences(session_factory=db.session, output_path=output_path, **filters)

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
        parser.add_argument("--based-on", "-b", help="Train on top of existing adapter (path)")

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
            ("fp16", "fp16"),
            ("bf16", "bf16"),
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

    def _build_configs(self) -> tuple[float, bool, RunConfig]:
        """Build training configuration from args and profile."""
        params: dict[str, Any] = {}
        beta, quantize = 0.1, True

        profile_name = getattr(self.args, "profile", None)
        if profile_name:
            beta, quantize, params = self._apply_profile(self._load_profile(profile_name))

        beta, quantize, params = self._apply_cli_overrides(beta, quantize, params)
        return beta, quantize, RunConfig(**params) if params else RunConfig()

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

    def _execute_training(  # cq: exempt=33
        self, data_path: Path, output_dir: Path, model_path: Path
    ) -> int:
        """Execute the DPO training."""
        from ...training.dpo import train_dpo

        beta, quantize, training_config = self._build_configs()
        based_on_arg = getattr(self.args, "based_on", None)
        based_on = Path(based_on_arg) if based_on_arg else None

        self.lg.info(
            "starting DPO training",
            extra={
                "data": str(data_path),
                "output": str(output_dir),
                "model": str(model_path),
                "beta": beta,
                "quantize": quantize,
                "epochs": training_config.num_epochs,
                "based_on": str(based_on) if based_on else None,
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
            based_on=based_on,
        )

        self._print_result(result)
        return 0

    def _print_result(self, result: RunResult) -> None:
        """Print training result summary."""
        print("\nTraining complete!")
        print(f"  Adapter: {result.adapter_path}")
        print(f"  Base model: {result.base_model}")
        print(f"  Duration: {result.duration_seconds:.1f}s")
        print(f"  Samples trained: {result.samples_trained}")
        _print_adapter_metadata(result.adapter_path)
        if result.metrics:
            print("  Metrics:")
            for key, value in result.metrics.items():
                if isinstance(value, float):
                    print(f"    {key}: {value:.4f}")
                else:
                    print(f"    {key}: {value}")


class SftTool(ModelResolutionMixin, Tool):
    """Train SFT adapter from feedback data."""

    def __init__(self, parent: Any = None) -> None:
        config = ToolConfig(
            name="sft",
            aliases=["s"],
            help_text="Train SFT adapter from feedback data",
        )
        super().__init__(parent, config)

    def add_args(self, parser) -> None:
        parser.add_argument("--data", "-d", help="Input JSONL path")
        parser.add_argument("--output", "-o", help="Output adapter directory")
        parser.add_argument("--model", "-m", help="Model name (from models.yaml)")
        parser.add_argument("--list-models", action="store_true", help="List available models")
        parser.add_argument("--no-quantize", action="store_true", help="Disable 4-bit quantization")
        parser.add_argument("--epochs", type=int, help="Number of training epochs")
        parser.add_argument("--batch-size", type=int, help="Training batch size")
        parser.add_argument("--lr", type=float, help="Learning rate")
        parser.add_argument("--profile", help="Use named training profile from config")
        parser.add_argument("--based-on", "-b", help="Train on top of existing adapter (path)")

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

    def _apply_profile(self, profile: dict) -> tuple[bool, dict[str, Any]]:
        """Apply profile settings, returning (quantize, params)."""
        params: dict[str, Any] = {}
        for key, param in [
            ("epochs", "num_epochs"),
            ("batch_size", "batch_size"),
            ("learning_rate", "learning_rate"),
            ("fp16", "fp16"),
            ("bf16", "bf16"),
        ]:
            if key in profile:
                params[param] = profile[key]
        return profile.get("quantize", True), params

    def _apply_cli_overrides(self, quantize: bool, params: dict) -> tuple[bool, dict]:
        """Apply CLI arg overrides to profile settings."""
        if getattr(self.args, "no_quantize", False):
            quantize = False
        for arg, param in [
            ("epochs", "num_epochs"),
            ("batch_size", "batch_size"),
            ("lr", "learning_rate"),
        ]:
            if getattr(self.args, arg, None) is not None:
                params[param] = getattr(self.args, arg)
        return quantize, params

    def _build_configs(self) -> tuple[bool, RunConfig]:
        """Build training configuration from args and profile."""
        params: dict[str, Any] = {}
        quantize = True

        profile_name = getattr(self.args, "profile", None)
        if profile_name:
            quantize, params = self._apply_profile(self._load_profile(profile_name))

        quantize, params = self._apply_cli_overrides(quantize, params)
        return quantize, RunConfig(**params) if params else RunConfig()

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

    def _resolve_based_on(self) -> Path | None:
        """Resolve --based-on argument to adapter path."""
        based_on_arg = getattr(self.args, "based_on", None)
        return Path(based_on_arg) if based_on_arg else None

    def _execute_training(self, data_path: Path, output_dir: Path, model_path: Path) -> int:
        """Execute the SFT training."""
        from ...training.lora import train_lora

        quantize, training_config = self._build_configs()
        based_on = self._resolve_based_on()
        self._log_training_start(
            data_path, output_dir, model_path, quantize, training_config, based_on
        )

        result = train_lora(
            lg=self.lg,
            data_path=data_path,
            output_dir=output_dir,
            base_model=str(model_path),
            lora_config=LoraConfig(),
            training_config=training_config,
            quantize=quantize,
            resume_from=based_on,
        )
        self._print_result(result)
        return 0

    def _log_training_start(
        self,
        data_path: Path,
        output_dir: Path,
        model_path: Path,
        quantize: bool,
        training_config: RunConfig,
        based_on: Path | None,
    ) -> None:
        """Log training start with parameters."""
        self.lg.info(
            "starting SFT training",
            extra={
                "data": str(data_path),
                "output": str(output_dir),
                "model": str(model_path),
                "quantize": quantize,
                "epochs": training_config.num_epochs,
                "based_on": str(based_on) if based_on else None,
            },
        )

    def _print_result(self, result: RunResult) -> None:
        """Print training result summary."""
        print("\nTraining complete!")
        print(f"  Adapter: {result.adapter_path}")
        print(f"  Base model: {result.base_model}")
        print(f"  Duration: {result.duration_seconds:.1f}s")
        print(f"  Samples trained: {result.samples_trained}")
        _print_adapter_metadata(result.adapter_path)
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

    def _build_training_result(self, adapter_path: Path) -> RunResult:
        """Build a minimal RunResult for registration."""

        return RunResult(
            adapter_path=adapter_path,
            base_model="unknown",
            method="dpo",
            metrics={},
            config={},
            started_at=utc_now(),
            completed_at=utc_now(),
            samples_trained=0,
        )

    def _do_register(self, adapter_path: Path, adapter_id: str, deploy: bool) -> int:
        """Execute registration and print result."""

        registry = AdapterRegistry(self.lg, self._get_base_path())
        training_result = self._build_training_result(adapter_path)

        self.lg.info("registering adapter", extra={"adapter_id": adapter_id, "deploy": deploy})

        try:
            info = registry.register(
                training_result=training_result,
                adapter_id=adapter_id,
                description=getattr(self.args, "description", None),
                deploy=deploy,
                overwrite=getattr(self.args, "overwrite", False),
            )
        except ValueError as e:
            self.lg.error("registration failed", extra={"error": str(e)})
            return 1

        status = "deployed" if info.deployed else "not deployed"
        print(f"Registered adapter '{info.adapter_id}' to {info.path}")
        print(f"  Status: {status}")
        _print_adapter_metadata(info.path)
        return 0

    def run(self, **kwargs: Any) -> int:
        adapter_path = Path(self.args.adapter)
        if not adapter_path.exists():
            self.lg.error("adapter path not found", extra={"path": str(adapter_path)})
            return 1

        deploy = not getattr(self.args, "disabled", False)
        return self._do_register(adapter_path, self.args.id, deploy)


class PipelineTool(ModelResolutionMixin, Tool):
    """Run full training workflow: export -> train -> register.

    Supports both DPO and SFT training methods. The method is determined by the
    queued run's `method` field in training_runs.
    """

    def __init__(self, parent: Any = None) -> None:
        config = ToolConfig(
            name="pipeline",
            aliases=["p"],
            help_text="Run training workflow: export -> train -> register (DPO or SFT)",
        )
        super().__init__(parent, config)
        self._db: Database | None = None
        self._learn_client = None
        self._run_id: int | None = None
        # Resolved from queue when --context/--id not provided
        self._resolved_context: str | None = None
        self._resolved_adapter_id: str | None = None
        self._run_method: str = "dpo"  # Default, updated from queue

    def add_args(self, parser) -> None:
        self._add_export_args(parser)
        self._add_training_args(parser)
        self._add_register_args(parser)

    def _add_export_args(self, parser) -> None:
        """Add export-related arguments."""
        parser.add_argument("--context", help="Context key for export")
        parser.add_argument("--category", help="Category filter for export")
        parser.add_argument("--min-margin", type=float, help="Minimum margin threshold")

    def _add_training_args(self, parser) -> None:
        """Add training-related arguments."""
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
        parser.add_argument(
            "--based-on", "-b", help="Train on top of existing adapter (path or ID)"
        )
        parser.add_argument("--run-id", type=int, help="Use existing run ID")
        parser.add_argument(
            "--retry", action="store_true", help="With --run-id, retry a completed run"
        )

    def _add_register_args(self, parser) -> None:
        """Add registration-related arguments."""
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

    def _get_database(self) -> Database:
        """Get or create cached Database instance."""
        if self._db is None:
            config = DotDict(**dict(self.app.config)) if self.app.config else DotDict()
            dbs = getattr(config, "dbs", None)
            if dbs is None or not hasattr(dbs, "main"):
                raise ValueError("Database not configured: missing dbs.main in config")
            self._db = Database(self.lg, PG(self.lg, dbs.main))
        return self._db

    def _get_learn_client(self):
        """Get or create LearnClient instance."""
        if self._learn_client is None:
            from llm_learn import LearnClient
            from llm_learn.memory import IsolationContext

            context_key = self._resolved_context or self.args.context
            context = IsolationContext(context_key=context_key)
            self._learn_client = LearnClient(
                lg=self.lg,
                database=self._get_database(),
                context=context,
                ensure_schema=False,
            )
        return self._learn_client

    def _get_dpo_client(self):
        """Get DpoClient via LearnClient."""
        return self._get_learn_client().train.dpo

    def _get_sft_client(self):
        """Get SftClient via LearnClient."""
        return self._get_learn_client().train.sft

    def _get_training_client(self):
        """Get appropriate training client based on run method."""
        if self._run_method == "sft":
            return self._get_sft_client()
        return self._get_dpo_client()

    @property
    def _adapter_id(self) -> str:
        """Get resolved adapter ID."""
        return self._resolved_adapter_id or self.args.id

    def _find_pending_run(
        self, context_arg: str | None
    ) -> tuple[int, str | None, str | None, str] | None:
        """Find earliest pending training run (DPO or SFT).

        Returns (id, context_key, adapter_name, method) or None.
        """
        from sqlalchemy import select

        from llm_learn.training.models import Run

        with self._get_database().session() as session:
            stmt = (
                select(Run)
                .where(Run.method.in_(["dpo", "sft"]))
                .where(Run.status == "pending")
                .where(_not_deleted_filter(Run))
                .order_by(Run.created_at.asc())
            )
            if context_arg:
                stmt = stmt.where(Run.context_key == context_arg)
            run = session.scalar(stmt)
            if run is None:
                return None
            # Extract values while still in session
            adapter_name = run.adapter.get("name") if run.adapter else None
            return (run.id, run.context_key, adapter_name, run.method)

    def _resolve_from_queue(self) -> None:
        """Resolve context, adapter_id, and method from pending queue if not provided."""
        context_arg = getattr(self.args, "context", None)
        id_arg = getattr(self.args, "id", None)

        # If both provided, nothing to resolve (default to dpo method for backward compat)
        if context_arg and id_arg:
            self._resolved_context = context_arg
            self._resolved_adapter_id = id_arg
            return

        result = self._find_pending_run(context_arg)
        if result is None:
            if context_arg:
                raise ValueError(f"No pending training runs found for context '{context_arg}'")
            raise ValueError("No pending training runs in queue")

        self._run_id, self._resolved_context, self._resolved_adapter_id, self._run_method = result
        if self._resolved_adapter_id is None:
            raise ValueError(f"Pending run {self._run_id} has no adapter name; use --id")
        self.lg.info(
            "resolved run from queue",
            extra={
                "run_id": self._run_id,
                "context": self._resolved_context,
                "method": self._run_method,
            },
        )

    def _get_or_create_run(self) -> int:
        """Get existing run or create new one."""
        # If already resolved from queue, use that
        if self._run_id is not None:
            return self._run_id

        client = self._get_training_client()

        # Use existing run if specified via --run-id
        if run_id := getattr(self.args, "run_id", None):
            run = client.get(run_id)
            if run is None:
                raise ValueError(f"Training run {run_id} not found")
            if run.status != "pending":
                raise ValueError(f"Training run {run_id} is {run.status}, expected pending")
            self._run_method = run.method
            return int(run_id)

        # Look for existing pending run with same adapter name
        pending = client.list_runs(status="pending")
        for run in pending:
            if run.adapter_name == self._adapter_id:
                self.lg.info("using existing pending run", extra={"run_id": run.id})
                return int(run.id)

        # Create new run
        run = client.create(adapter_name=self._adapter_id)
        self.lg.info(f"created new {self._run_method.upper()} run", extra={"run_id": run.id})
        return int(run.id)

    def _run_export(self, output_path: Path) -> int:
        """Run export step - exports pending data for this run (pairs for DPO, examples for SFT)."""
        assert self._run_id is not None, "_run_id must be set before export"

        if self._run_method == "sft":
            return self._run_export_sft(output_path)
        return self._run_export_dpo(output_path)

    def _run_export_dpo(self, output_path: Path) -> int:
        """Export pending DPO pairs."""
        assert self._run_id is not None
        self.lg.info("step 1/3: exporting pairs", extra={"run_id": self._run_id})
        result = export_run_pairs(
            session_factory=self._get_database().session,
            run_id=self._run_id,
            output_path=output_path,
        )
        if result.count == 0:
            self.lg.error("no pairs found for run")
            return 1
        print(f"[1/3] Exported {result.count} pairs")
        return 0

    def _run_export_sft(self, output_path: Path) -> int:
        """Export pending SFT examples."""
        assert self._run_id is not None
        self.lg.info("step 1/3: exporting examples", extra={"run_id": self._run_id})
        result = export_run_examples(
            session_factory=self._get_database().session,
            run_id=self._run_id,
            output_path=output_path,
        )
        if result.count == 0:
            self.lg.error("no examples found for run")
            return 1
        print(f"[1/3] Exported {result.count} examples")
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
            ("fp16", "fp16"),
            ("bf16", "bf16"),
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

    def _build_training_params(self) -> tuple[float, bool, RunConfig]:
        """Build training parameters from profile and CLI args."""
        params: dict[str, Any] = {}
        beta, quantize = 0.1, True

        if profile_name := getattr(self.args, "profile", None):
            beta, quantize, params = self._load_profile(profile_name)

        beta, quantize, params = self._apply_cli_overrides(beta, quantize, params)

        # When quantizing, use bf16 (matches bnb_4bit_compute_dtype=torch.bfloat16)
        if quantize and "fp16" not in params and "bf16" not in params:
            params["fp16"] = False
            params["bf16"] = True

        return beta, quantize, RunConfig(**params) if params else RunConfig()

    def _resolve_based_on(self) -> Path | None:
        """Resolve adapter to train on top of.

        Priority:
        1. CLI --based-on argument (explicit override)
        2. DB run's based_on field (cross-method lineage from parent run)
        """
        # 1. Check CLI argument first (explicit override)
        based_on_arg = getattr(self.args, "based_on", None)
        if based_on_arg:
            return self._resolve_adapter_path(based_on_arg)

        # 2. Check DB run's based_on field (lineage from parent run)
        if self._run_id is not None:
            adapter_name = self._get_parent_adapter_name()
            if adapter_name:
                self.lg.debug(
                    "resolved based_on from DB lineage",
                    extra={"parent_adapter": adapter_name},
                )
                return self._resolve_adapter_path(adapter_name)

        return None

    def _get_parent_adapter_name(self) -> str | None:
        """Get parent run's adapter name from DB lineage."""
        from ...training.models import Run

        with self._get_database().session() as session:
            # Get current run's based_on
            current = session.get(Run, self._run_id)
            if current is None or current.based_on is None:
                return None

            # Get parent run's adapter name
            parent = session.get(Run, current.based_on)
            if parent is None or parent.adapter is None:
                return None

            return parent.adapter.get("name")

    def _resolve_adapter_path(self, adapter_ref: str) -> Path:
        """Resolve adapter reference (name or path) to actual path."""
        base_path = self._get_adapter_base_path()
        if base_path is None:
            # No registry configured, try as raw path
            path = Path(adapter_ref)
            if not path.exists():
                raise ValueError(f"Adapter not found: {adapter_ref}")
            return path

        registry = AdapterRegistry(self.lg, base_path)
        return registry.resolve(adapter_ref)

    def _run_train(self, data_path: Path, adapter_path: Path, model_path: Path) -> RunResult | None:
        """Run training step - dispatches to DPO or SFT trainer based on method."""
        if self._run_method == "sft":
            return self._run_train_sft(data_path, adapter_path, model_path)
        return self._run_train_dpo(data_path, adapter_path, model_path)

    def _run_train_dpo(
        self, data_path: Path, adapter_path: Path, model_path: Path
    ) -> RunResult | None:
        """Run DPO training."""
        from ...training.dpo import train_dpo

        beta, quantize, training_config = self._build_training_params()
        based_on = self._resolve_based_on()
        self._log_train_step("DPO", model_path, training_config, based_on)

        try:
            result: RunResult = train_dpo(
                lg=self.lg,
                data_path=data_path,
                output_dir=adapter_path,
                base_model=str(model_path),
                lora_config=LoraConfig(),
                training_config=training_config,
                beta=beta,
                quantize=quantize,
                based_on=based_on,
            )
            return self._handle_train_success(result)
        except Exception as e:
            self.lg.error("DPO training failed", extra={"exception": e})
            return None

    def _run_train_sft(
        self, data_path: Path, adapter_path: Path, model_path: Path
    ) -> RunResult | None:
        """Run SFT training."""
        from ...training.lora import train_lora

        _, quantize, training_config = self._build_training_params()
        based_on = self._resolve_based_on()
        self._log_train_step("SFT", model_path, training_config, based_on)

        try:
            result: RunResult = train_lora(
                lg=self.lg,
                data_path=data_path,
                output_dir=adapter_path,
                base_model=str(model_path),
                lora_config=LoraConfig(),
                training_config=training_config,
                quantize=quantize,
                resume_from=based_on,
            )
            return self._handle_train_success(result)
        except Exception as e:
            self.lg.error("SFT training failed", extra={"exception": e})
            return None

    def _log_train_step(
        self, method: str, model_path: Path, config: RunConfig, based_on: Path | None
    ) -> None:
        """Log training step start."""
        self.lg.info(
            f"step 2/3: training {method} adapter",
            extra={
                "model": str(model_path),
                "epochs": config.num_epochs,
                "based_on": str(based_on) if based_on else None,
            },
        )

    def _handle_train_success(self, result: RunResult) -> RunResult:
        """Handle successful training - print summary and return result."""
        print(f"[2/3] Training complete ({result.duration_seconds:.1f}s)")
        _print_adapter_metadata(result.adapter_path, label="Trained adapter")
        return result

    def _get_adapter_base_path(self) -> str | None:
        """Get adapter base path from config."""
        config = DotDict(**dict(self.app.config)) if self.app.config else DotDict()
        adapters_cfg = getattr(config, "adapters", None)
        lora_cfg = getattr(adapters_cfg, "lora", None) if adapters_cfg else None
        return lora_cfg.base_path if lora_cfg and hasattr(lora_cfg, "base_path") else None

    def _do_register(self, registry, training_result: RunResult, overwrite: bool, deploy: bool):
        """Perform the actual adapter registration."""
        return registry.register(
            training_result=training_result,
            adapter_id=self._adapter_id,
            description=getattr(self.args, "description", None),
            deploy=deploy,
            overwrite=overwrite,
        )

    def _confirm_overwrite(self, registry) -> bool | None:
        """Check if adapter exists and confirm overwrite. Returns None to skip."""
        if getattr(self.args, "overwrite", False):
            return True
        existing = registry.get(self._adapter_id)
        if not existing:
            return False
        response = input(f"Adapter '{self._adapter_id}' already exists. Overwrite? [y/N] ")
        if response.strip().lower() not in ("y", "yes"):
            print("Registration skipped.")
            return None
        return True

    def _run_register(self, training_result: RunResult) -> int:
        """Run registration step with interactive prompts."""
        response = input("\nCopy to registry? [Y/n] ").strip().lower()
        if response in ("n", "no"):
            print(f"Adapter available at: {training_result.adapter_path}")
            return 0

        base_path = self._get_adapter_base_path()
        if base_path is None:
            self.lg.error("adapters.lora.base_path not configured")
            return 1

        registry = AdapterRegistry(self.lg, base_path)
        overwrite = self._confirm_overwrite(registry)
        if overwrite is None:
            return 0

        response = input("Deploy adapter? [Y/n] ").strip().lower()
        deploy = response not in ("n", "no")

        try:
            info = self._do_register(registry, training_result, overwrite, deploy)
            status = "and deployed" if deploy else "(not deployed)"
            print(f"Registered {status} adapter '{info.adapter_id}'")
            _print_adapter_metadata(info.path, label="Adapter")
            return 0
        except ValueError as e:
            self.lg.error("registration failed", extra={"error": str(e)})
            return 1

    def _setup_paths(self, adapter_id: str) -> tuple[Path, Path, Path]:
        """Set up working directory and paths."""
        output_dir = self._get_output_dir()
        work_dir = output_dir / adapter_id
        work_dir.mkdir(parents=True, exist_ok=True)
        return work_dir, work_dir / "data.jsonl", work_dir / "adapter"

    def run(self, **kwargs: Any) -> int:
        if getattr(self.args, "list_models", False):
            return self._list_models()

        model_path = self._resolve_model(getattr(self.args, "model", None))
        if model_path is None:
            return 1

        try:
            # Check for --run with --retry (creates new run from completed one)
            run_id = getattr(self.args, "run_id", None)
            is_retry = getattr(self.args, "retry", False)
            if run_id and is_retry:
                self._setup_retry_run(run_id)
            else:
                self._resolve_from_queue()
                self._run_id = self._get_or_create_run()
        except ValueError as e:
            self.lg.error(str(e))
            return 1

        work_dir, data_path, adapter_path = self._setup_paths(self._adapter_id)
        self._print_pipeline_header(model_path, work_dir)
        return self._execute_pipeline(data_path, adapter_path, model_path)

    def _setup_retry_run(self, original_run_id: int) -> None:
        """Set up a new run by copying pairs from a completed run.

        Used with --run ID --retry to retrain with different hyperparameters
        without re-creating pairs on the agent side.
        """
        from ...training.models import Run

        with self._get_database().session() as session:
            original = self._validate_retry_source(session, original_run_id, Run)
            new_run_id, copied = self._create_retry_run(session, original, Run)

            self.lg.info(
                f"created retry run from #{original_run_id}",
                extra={"new_run_id": new_run_id, "copied": copied, "method": original.method},
            )
            self._run_id = new_run_id

    def _validate_retry_source(self, session, run_id: int, run_cls):
        """Validate and return the source run for retry."""
        original = session.get(run_cls, run_id)
        if original is None:
            raise ValueError(f"Run {run_id} not found")
        if original.status != "completed":
            raise ValueError(f"Run {run_id} is {original.status}, expected completed")
        if original.method not in ("dpo", "sft"):
            raise ValueError(f"Unsupported method: {original.method}")

        adapter_name = original.adapter.get("name") if original.adapter else None
        if not adapter_name:
            raise ValueError(f"Run {run_id} has no adapter name")

        self._run_method = original.method
        self._resolved_context = original.context_key
        self._resolved_adapter_id = adapter_name
        return original

    def _create_retry_run(self, session, original, run_cls) -> tuple[int, int]:
        """Create new run and copy pairs/examples from original."""
        new_run = run_cls(
            method=original.method,
            context_key=original.context_key,
            adapter={"name": self._resolved_adapter_id},
            based_on=original.based_on,
            status="pending",
        )
        session.add(new_run)
        session.flush()

        if original.method == "dpo":
            copied = self._copy_dpo_pairs(session, original.id, new_run.id)
        else:
            copied = self._copy_sft_examples(session, original.id, new_run.id)

        return new_run.id, copied

    def _copy_dpo_pairs(self, session, source_run_id: int, target_run_id: int) -> int:
        """Copy trained DPO pairs to pending for new run."""
        from sqlalchemy import insert, select

        from ...training.dpo.client import PendingPair, TrainedPair

        # Get trained pairs from source
        stmt = select(TrainedPair).where(TrainedPair.run_id == source_run_id)
        trained = session.execute(stmt).scalars().all()

        if not trained:
            raise ValueError(f"Run {source_run_id} has no trained pairs")

        # Insert as pending for target
        for pair in trained:
            session.execute(
                insert(PendingPair).values(
                    run_id=target_run_id,
                    chosen_fact_id=pair.chosen_fact_id,
                    rejected_fact_id=pair.rejected_fact_id,
                    prompt=pair.prompt,
                )
            )

        return len(trained)

    def _copy_sft_examples(self, session, source_run_id: int, target_run_id: int) -> int:
        """Copy trained SFT examples to pending for new run."""
        from sqlalchemy import insert, select

        from ...training.sft.client import PendingExample, TrainedExample

        # Get trained examples from source
        stmt = select(TrainedExample).where(TrainedExample.run_id == source_run_id)
        trained = session.execute(stmt).scalars().all()

        if not trained:
            raise ValueError(f"Run {source_run_id} has no trained examples")

        # Insert as pending for target
        for ex in trained:
            session.execute(
                insert(PendingExample).values(
                    run_id=target_run_id,
                    fact_id=ex.fact_id,
                )
            )

        return len(trained)

    def _print_pipeline_header(self, model_path: Path, work_dir: Path) -> None:
        """Print pipeline startup info."""
        method_label = self._run_method.upper()
        print(f"\n{method_label} Pipeline: {self._resolved_context} -> {self._adapter_id}")
        print(f"Model: {model_path.name}")
        print(f"Output directory: {work_dir}")
        print(f"Run ID: {self._run_id}\n")

    def _execute_pipeline(self, data_path: Path, adapter_path: Path, model_path: Path) -> int:
        """Execute the three pipeline steps with run tracking."""
        client = self._get_training_client()

        data_type = "examples" if self._run_method == "sft" else "preferences"
        if self._run_export(data_path) != 0:
            client.fail(self._run_id, f"Export failed: no {data_type} found")
            return self._pipeline_fail("step 1 (export)")

        client.start(self._run_id)
        training_result = self._run_train(data_path, adapter_path, model_path)
        if training_result is None:
            # Reset to pending - training crash is often transient (OOM, etc.)
            client.reset(self._run_id)
            return self._pipeline_fail("step 2 (training) - run reset to pending for retry")

        return self._finish_pipeline(client, training_result)

    def _pipeline_fail(self, step: str) -> int:
        """Print pipeline failure message and return error code."""
        print(f"\nPipeline failed at {step}")
        return 1

    def _finish_pipeline(self, client, training_result: RunResult) -> int:
        """Handle registration and completion after successful training."""
        skip_register = getattr(self.args, "skip_register", False)
        reg_failed = False if skip_register else self._run_register(training_result) != 0

        # Record metrics and adapter info
        metrics, adapter_info = self._extract_run_data(training_result)
        client.complete(self._run_id, metrics=metrics, adapter_info=adapter_info)

        if skip_register:
            print(f"\nTraining complete! Adapter at: {training_result.adapter_path}")
        elif reg_failed:
            print("\nTraining complete, but registration failed (see above)")
            return 1
        else:
            print(f"\nPipeline complete! Adapter '{self._adapter_id}' is ready.")
        return 0

    def _extract_run_data(self, result: RunResult) -> tuple[dict, dict]:
        """Extract metrics and adapter info from training result."""
        metrics = dict(result.metrics) if result.metrics else {}
        metrics["duration_seconds"] = result.duration_seconds
        metrics["samples_trained"] = result.samples_trained

        adapter_meta = compute_adapter_metadata(result.adapter_path)
        adapter_info = {"mtime": adapter_meta.mtime, "md5": adapter_meta.md5}
        return metrics, adapter_info


class ResetTool(Tool):
    """Reset training data (DPO and SFT) for a context.

    Soft-deletes runs and clears all associated data (pending and trained).
    This frees examples/pairs for reuse in future training runs.
    """

    # Display name for NULL context_key (used in list output)
    _NULL_CONTEXT_DISPLAY = "(no context)"

    # Sentinel for "all contexts" in reset --all
    _ALL_CONTEXTS: object = object()

    def __init__(self, parent: Any = None) -> None:
        config = ToolConfig(
            name="reset",
            help_text="Reset training runs for a context (DPO and SFT)",
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

    def _context_filter(self, column, context_key: str | None | object):
        """Build SQLAlchemy filter for context_key (handles NULL and all-contexts)."""
        if context_key is self._ALL_CONTEXTS:
            return literal(True)  # Match all rows
        return column.is_(None) if context_key is None else column == context_key

    def _count_items(self, session, table, method: str, context_filter) -> int:
        """Count items in a training data table for a given method and context."""
        return (
            session.scalar(
                select(func.count())
                .select_from(table)
                .join(Run, table.run_id == Run.id)
                .where(context_filter, Run.method == method, _not_deleted_filter(Run))
            )
            or 0
        )

    def _count_data(self, session, context_key: str | None | object) -> dict[str, int]:
        """Count all training data for a context."""
        ctx = self._context_filter(Run.context_key, context_key)
        run_count = (
            session.scalar(
                select(func.count())
                .select_from(Run)
                .where(ctx, Run.method.in_(["dpo", "sft"]), _not_deleted_filter(Run))
            )
            or 0
        )
        return {
            "runs": run_count,
            "dpo_pending": self._count_items(session, PendingPair, "dpo", ctx),
            "sft_pending": self._count_items(session, PendingExample, "sft", ctx),
            "dpo_trained": self._count_items(session, TrainedPair, "dpo", ctx),
            "sft_trained": self._count_items(session, TrainedExample, "sft", ctx),
        }

    def _list_contexts(self, session) -> list[tuple[str, int, int]]:
        """List all contexts with run counts and total data counts."""
        stmt = (
            select(Run.context_key, func.count(Run.id).label("run_count"))
            .where(Run.method.in_(["dpo", "sft"]), _not_deleted_filter(Run))
            .group_by(Run.context_key)
            .order_by(Run.context_key)
        )
        results = []
        for row in session.execute(stmt):
            display_key = row.context_key or self._NULL_CONTEXT_DISPLAY
            counts = self._count_data(session, row.context_key)
            total = (
                counts["dpo_pending"]
                + counts["sft_pending"]
                + counts["dpo_trained"]
                + counts["sft_trained"]
            )
            results.append((display_key, row.run_count, total))
        return results

    def _empty_result(self) -> dict[str, int]:
        """Return empty result dict for when there's nothing to delete."""
        return {"runs": 0, "dpo_pending": 0, "sft_pending": 0, "dpo_trained": 0, "sft_trained": 0}

    def _delete_items(self, session, table, run_ids: list[int]) -> int:
        """Delete items from a training data table for given run IDs."""
        return int(session.execute(delete(table).where(table.run_id.in_(run_ids))).rowcount)

    def _delete_data(self, session, context_key: str | None | object) -> dict[str, int]:
        """Delete runs and all associated data. Returns counts of deleted items."""
        ctx = self._context_filter(Run.context_key, context_key)
        runs_stmt = select(Run).where(ctx, Run.method.in_(["dpo", "sft"]), _not_deleted_filter(Run))
        runs = list(session.scalars(runs_stmt).all())

        if not runs:
            return self._empty_result()

        run_ids = [r.id for r in runs]
        result = {
            "dpo_pending": self._delete_items(session, PendingPair, run_ids),
            "sft_pending": self._delete_items(session, PendingExample, run_ids),
            "dpo_trained": self._delete_items(session, TrainedPair, run_ids),
            "sft_trained": self._delete_items(session, TrainedExample, run_ids),
            "runs": len(runs),
        }
        now = utc_now().isoformat()
        for run in runs:
            run.system_status = {"deleted": True, "deleted_at": now}
        return result

    def run(self, **kwargs: Any) -> int:
        context_key = getattr(self.args, "context", None)
        reset_all = getattr(self.args, "all", False)
        confirm = getattr(self.args, "confirm", False)

        db = self._get_database()

        with db.session() as session:
            if reset_all:
                return self._run_reset_all(session, confirm)
            if not context_key:
                return self._run_list_contexts(session)
            return self._run_reset_context(session, context_key, confirm)

    def _run_list_contexts(self, session) -> int:
        """List available contexts."""
        contexts = self._list_contexts(session)
        if not contexts:
            print("No training runs found.")
            return 0

        print("\nAvailable contexts:\n")
        print(f"  {'Context':<30} {'Runs':>8} {'Data':>8}")
        print(f"  {'-' * 30} {'-' * 8} {'-' * 8}")
        for ctx, runs, total_data in contexts:
            print(f"  {ctx:<30} {runs:>8} {total_data:>8}")
        print("\nUse --context <name> to reset a specific context.")
        return 0

    def _print_counts(self, counts: dict[str, int]) -> None:
        """Print data counts for confirmation."""
        print(f"  Training runs:     {counts['runs']} (soft-delete)")
        print(f"  DPO pending pairs: {counts['dpo_pending']}")
        print(f"  DPO trained pairs: {counts['dpo_trained']}")
        print(f"  SFT pending:       {counts['sft_pending']}")
        print(f"  SFT trained:       {counts['sft_trained']}")

    def _print_results(self, result: dict[str, int]) -> None:
        """Print deletion results."""
        parts = [f"soft-deleted {result['runs']} runs"]
        pending = result["dpo_pending"] + result["sft_pending"]
        trained = result["dpo_trained"] + result["sft_trained"]
        if pending:
            parts.append(f"cleared {pending} pending")
        if trained:
            parts.append(f"deleted {trained} trained")
        print(", ".join(parts))

    def _run_reset_context(self, session, context_key: str, confirm: bool) -> int:
        """Reset a specific context."""
        db_context_key = self._normalize_context_key(context_key)
        counts = self._count_data(session, db_context_key)

        if counts["runs"] == 0:
            print(f"No training data found for context '{context_key}'")
            return 0

        print(f"\nThis will reset for context '{context_key}':")
        self._print_counts(counts)
        print()

        if not confirm:
            response = input("Proceed? [y/N] ").strip().lower()
            if response != "y":
                print("Aborted.")
                return 1

        result = self._delete_data(session, db_context_key)
        session.commit()

        self._print_results(result)
        return 0

    def _run_reset_all(self, session, confirm: bool) -> int:
        """Reset all contexts."""
        counts = self._count_data(session, self._ALL_CONTEXTS)

        if counts["runs"] == 0:
            print("No training data found.")
            return 0

        print("\nThis will reset ALL training data:")
        self._print_counts(counts)
        print()

        if not confirm:
            response = input("Are you sure? [y/N] ").strip().lower()
            if response != "y":
                print("Aborted.")
                return 1

        result = self._delete_data(session, self._ALL_CONTEXTS)
        session.commit()

        self._print_results(result)
        return 0


class TrainTool(Tool):
    """Training commands for DPO and SFT workflows."""

    def __init__(self, parent: Any = None) -> None:
        config = ToolConfig(
            name="train",
            aliases=["t"],
            help_text="Training commands (export, dpo, sft, register, pipeline)",
        )
        super().__init__(parent, config)
        self.add_tool(ExportTool(self))
        self.add_tool(DpoTool(self))
        self.add_tool(SftTool(self))
        self.add_tool(RegisterTool(self))
        self.add_tool(PipelineTool(self))
        self.add_tool(ResetTool(self))

    def run(self, **kwargs: Any) -> int:
        """Delegate to subtool."""
        result: int = self.group.run(**kwargs)
        return result
