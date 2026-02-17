"""Training tools - CLI for DPO training workflow."""

from datetime import datetime
from pathlib import Path
from typing import Any

from appinfra.app.tools import Tool, ToolConfig
from appinfra.db.pg import PG
from appinfra.dot_dict import DotDict
from llm_infer import compute_adapter_metadata
from llm_infer.models import ModelResolver
from sqlalchemy import delete, func, select

from ...core.database import Database
from ...core.utils import utc_now
from ...training.config import RunConfig, RunResult
from ...training.dpo import (
    PendingPair,
    Run,
    _not_deleted_filter,
    export_preferences,
    export_run_pairs,
)
from ...training.lora import AdapterRegistry
from ...training.lora import Config as LoraConfig


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
    """Run full DPO workflow: export -> train -> register."""

    def __init__(self, parent: Any = None) -> None:
        config = ToolConfig(
            name="pipeline",
            aliases=["p"],
            help_text="Run full DPO workflow: export -> train -> register",
        )
        super().__init__(parent, config)
        self._db: Database | None = None
        self._learn_client = None
        self._run_id: int | None = None
        # Resolved from queue when --context/--id not provided
        self._resolved_context: str | None = None
        self._resolved_adapter_id: str | None = None

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
        parser.add_argument(
            "--based-on", "-b", help="Train on top of existing adapter (path or ID)"
        )

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

    @property
    def _adapter_id(self) -> str:
        """Get resolved adapter ID."""
        return self._resolved_adapter_id or self.args.id

    def _find_pending_run(
        self, context_arg: str | None
    ) -> tuple[int, str | None, str | None] | None:
        """Find earliest pending DPO run. Returns (id, context_key, adapter_name) or None."""
        from sqlalchemy import select

        from llm_learn.training.dpo.client import Run

        with self._get_database().session() as session:
            stmt = (
                select(Run)
                .where(Run.method == "dpo")
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
            return (run.id, run.context_key, adapter_name)

    def _resolve_from_queue(self) -> None:
        """Resolve context and adapter_id from pending queue if not provided."""
        context_arg = getattr(self.args, "context", None)
        id_arg = getattr(self.args, "id", None)

        # If both provided, nothing to resolve
        if context_arg and id_arg:
            self._resolved_context = context_arg
            self._resolved_adapter_id = id_arg
            return

        result = self._find_pending_run(context_arg)
        if result is None:
            if context_arg:
                raise ValueError(f"No pending DPO runs found for context '{context_arg}'")
            raise ValueError("No pending DPO runs in queue")

        self._run_id, self._resolved_context, self._resolved_adapter_id = result
        self.lg.info(
            "resolved run from queue",
            extra={"run_id": self._run_id, "context": self._resolved_context},
        )

    def _get_or_create_run(self) -> int:
        """Get existing run or create new one."""
        # If already resolved from queue, use that
        if self._run_id is not None:
            return self._run_id

        client = self._get_dpo_client()

        # Use existing run if specified via --run-id
        if run_id := getattr(self.args, "run_id", None):
            run = client.get(run_id)
            if run is None:
                raise ValueError(f"DPO run {run_id} not found")
            if run.status != "pending":
                raise ValueError(f"DPO run {run_id} is {run.status}, expected pending")
            return int(run_id)

        # Look for existing pending run with same adapter name
        pending = client.list_runs(status="pending")
        for run in pending:
            if run.adapter_name == self._adapter_id:
                self.lg.info("using existing pending run", extra={"run_id": run.id})
                return int(run.id)

        # Create new run
        run = client.create(adapter_name=self._adapter_id)
        self.lg.info("created new DPO run", extra={"run_id": run.id})
        return int(run.id)

    def _run_export(self, output_path: Path) -> int:
        """Run export step - exports pending pairs for this run."""
        assert self._run_id is not None, "_run_id must be set before export"
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

    def _build_training_params(self) -> tuple[float, bool, RunConfig]:
        """Build training parameters from profile and CLI args."""
        params: dict[str, Any] = {}
        beta, quantize = 0.1, True

        if profile_name := getattr(self.args, "profile", None):
            beta, quantize, params = self._load_profile(profile_name)

        beta, quantize, params = self._apply_cli_overrides(beta, quantize, params)
        return beta, quantize, RunConfig(**params) if params else RunConfig()

    def _resolve_based_on(self) -> Path | None:
        """Resolve --based-on argument to adapter path."""
        based_on_arg = getattr(self.args, "based_on", None)
        if not based_on_arg:
            return None

        base_path = self._get_adapter_base_path()
        if base_path is None:
            # No registry configured, try as raw path
            path = Path(based_on_arg)
            if not path.exists():
                raise ValueError(f"Adapter not found: {based_on_arg}")
            return path

        registry = AdapterRegistry(self.lg, base_path)
        return registry.resolve(based_on_arg)

    def _run_train(  # cq: exempt=32
        self, data_path: Path, adapter_path: Path, model_path: Path
    ) -> RunResult | None:
        """Run training step."""
        from ...training.dpo import train_dpo

        beta, quantize, training_config = self._build_training_params()
        based_on = self._resolve_based_on()

        self.lg.info(
            "step 2/3: training DPO adapter",
            extra={
                "model": str(model_path),
                "epochs": training_config.num_epochs,
                "based_on": str(based_on) if based_on else None,
            },
        )

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
            print(f"[2/3] Training complete ({result.duration_seconds:.1f}s)")
            _print_adapter_metadata(result.adapter_path, label="Trained adapter")
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
            self._resolve_from_queue()
            self._run_id = self._get_or_create_run()
        except ValueError as e:
            self.lg.error(str(e))
            return 1

        work_dir, data_path, adapter_path = self._setup_paths(self._adapter_id)
        self._print_pipeline_header(model_path, work_dir)
        return self._execute_pipeline(data_path, adapter_path, model_path)

    def _print_pipeline_header(self, model_path: Path, work_dir: Path) -> None:
        """Print pipeline startup info."""
        print(f"\nDPO Pipeline: {self._resolved_context} -> {self._adapter_id}")
        print(f"Model: {model_path.name}")
        print(f"Output directory: {work_dir}")
        print(f"Run ID: {self._run_id}\n")

    def _execute_pipeline(self, data_path: Path, adapter_path: Path, model_path: Path) -> int:
        """Execute the three pipeline steps with run tracking."""
        client = self._get_dpo_client()

        if self._run_export(data_path) != 0:
            client.fail(self._run_id, "Export failed: no preferences found")
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
    """Reset DPO training data for a context.

    Clears pending pairs only - completed runs and their trained pairs remain intact.
    Uses soft-delete via system_status JSONB.
    """

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

    def _count_pending_pairs(self, session, context_key: str | None) -> int:
        """Count pending pairs for a context."""

        stmt = (
            select(func.count())
            .select_from(PendingPair)
            .join(Run, PendingPair.run_id == Run.id)
            .where(
                self._context_filter(Run.context_key, context_key),
                Run.method == "dpo",
                _not_deleted_filter(Run),
            )
        )
        return session.scalar(stmt) or 0

    def _list_contexts(self, session) -> list[tuple[str, int, int]]:
        """List all contexts with run and pending pair counts."""

        stmt = (
            select(Run.context_key, func.count(Run.id).label("run_count"))
            .where(Run.method == "dpo", _not_deleted_filter(Run))
            .group_by(Run.context_key)
            .order_by(Run.context_key)
        )
        results = []
        for row in session.execute(stmt):
            display_key = row.context_key or self._NULL_CONTEXT_DISPLAY
            pair_count = self._count_pending_pairs(session, row.context_key)
            results.append((display_key, row.run_count, pair_count))

        return results

    def _count_data(self, session, context_key: str | None) -> tuple[int, int]:
        """Count runs and pending pairs for the context."""

        context_filter = self._context_filter(Run.context_key, context_key)

        # Count runs (excluding soft-deleted)
        run_count_stmt = (
            select(func.count())
            .select_from(Run)
            .where(context_filter, Run.method == "dpo", _not_deleted_filter(Run))
        )
        run_count = session.scalar(run_count_stmt) or 0

        # Count pending pairs linked to those runs
        pair_count_stmt = (
            select(func.count())
            .select_from(PendingPair)
            .join(Run, PendingPair.run_id == Run.id)
            .where(context_filter, Run.method == "dpo", _not_deleted_filter(Run))
        )
        pair_count = session.scalar(pair_count_stmt) or 0

        return run_count, pair_count

    def _delete_data(self, session, context_key: str | None) -> tuple[int, int]:
        """Soft-delete runs and clear pending pairs. Returns (runs_deleted, pairs_freed)."""

        context_filter = self._context_filter(Run.context_key, context_key)

        # Get runs for this context (excluding already soft-deleted)
        runs_stmt = select(Run).where(context_filter, Run.method == "dpo", _not_deleted_filter(Run))
        runs = list(session.scalars(runs_stmt).all())

        if not runs:
            return 0, 0

        run_ids = [r.id for r in runs]

        # Delete pending pairs (hard delete - they become available for other runs)
        pairs_stmt = delete(PendingPair).where(PendingPair.run_id.in_(run_ids))
        pairs_result = session.execute(pairs_stmt)
        pairs_deleted = pairs_result.rowcount

        # Soft-delete runs via system_status
        now = utc_now().isoformat()
        for run in runs:
            run.system_status = {"deleted": True, "deleted_at": now}

        return len(runs), pairs_deleted

    def _count_all_data(self, session) -> tuple[int, int]:
        """Count all runs and pending pairs across all contexts."""

        run_count = (
            session.scalar(
                select(func.count())
                .select_from(Run)
                .where(Run.method == "dpo", _not_deleted_filter(Run))
            )
            or 0
        )
        # Only count pending pairs for active runs
        pair_count = (
            session.scalar(
                select(func.count())
                .select_from(PendingPair)
                .join(Run, PendingPair.run_id == Run.id)
                .where(Run.method == "dpo", _not_deleted_filter(Run))
            )
            or 0
        )
        return run_count, pair_count

    def _delete_all_data(self, session) -> tuple[int, int]:
        """Soft-delete all runs and clear pending pairs. Returns (runs_deleted, pairs_freed)."""

        # Get all non-deleted runs
        runs = list(
            session.scalars(select(Run).where(Run.method == "dpo", _not_deleted_filter(Run))).all()
        )

        if not runs:
            return 0, 0

        run_ids = [r.id for r in runs]

        # Delete pending pairs (hard delete)
        pairs_result = session.execute(delete(PendingPair).where(PendingPair.run_id.in_(run_ids)))
        pairs_deleted = pairs_result.rowcount

        # Soft-delete runs via system_status
        now = utc_now().isoformat()
        for run in runs:
            run.system_status = {"deleted": True, "deleted_at": now}

        return len(runs), pairs_deleted

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
        print(f"  {'Context':<30} {'Runs':>8} {'Pending':>10}")
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

        print(f"\nThis will reset for context '{context_key}':")
        print(f"  Training runs:    {run_count} (soft-delete)")
        print(f"  Pending pairs:    {pair_count} (cleared)")
        print("  Note: Completed runs and trained pairs are preserved.")
        print()

        if not confirm:
            response = input("Proceed? [y/N] ").strip().lower()
            if response != "y":
                print("Aborted.")
                return 1

        runs_deleted, pairs_deleted = self._delete_data(session, db_context_key)
        session.commit()

        print(f"Soft-deleted {runs_deleted} runs, cleared {pairs_deleted} pending pairs")
        return 0

    def _run_reset_all(self, session, confirm: bool) -> int:
        """Reset all contexts."""
        run_count, pair_count = self._count_all_data(session)

        if run_count == 0:
            print("No training data found.")
            return 0

        print("\nThis will reset ALL training data:")
        print(f"  Training runs:    {run_count} (soft-delete)")
        print(f"  Pending pairs:    {pair_count} (cleared)")
        print("  Note: Completed runs and trained pairs are preserved.")
        print()

        if not confirm:
            response = input("Are you sure? [y/N] ").strip().lower()
            if response != "y":
                print("Aborted.")
                return 1

        runs_deleted, pairs_deleted = self._delete_all_data(session)
        session.commit()

        print(f"Soft-deleted {runs_deleted} runs, cleared {pairs_deleted} pending pairs")
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
