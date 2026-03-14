"""Manifest loading and saving.

Handles YAML serialization of training manifests and data resolution.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from appinfra import DotDict

from ..schema import Adapter, RunResult
from .errors import CorruptedManifestError
from .schema import Data, Deployment, Manifest, Source


def _parse_datetime(value: str | datetime | None) -> datetime:
    """Parse datetime from string or pass through."""
    if value is None:
        return datetime.now(UTC)
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(value)


def _read_yaml_file(path: Path) -> Any:
    """Read YAML from file (supports .gz)."""
    import gzip

    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return yaml.safe_load(f)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _read_yaml_metadata(path: Path) -> Any:
    """Read YAML metadata only, stopping before data records.

    Streams gzip decompression and stops at 'data:' section to avoid
    decompressing hundreds of training records when only metadata is needed.
    """
    import gzip

    lines: list[str] = []
    opener = gzip.open if path.suffix == ".gz" else open

    with opener(path, "rt", encoding="utf-8") as f:  # type: ignore[call-overload]
        for line in f:
            # Stop when we hit the data section (top-level key, no indent)
            if line.startswith("data:"):
                break
            lines.append(line)

    return yaml.safe_load("".join(lines))


def _validate_required_fields(data: dict[str, Any]) -> None:
    """Validate required manifest fields."""
    if not data.get("adapter"):
        raise ValueError("Manifest missing required field: adapter")
    if not data.get("method"):
        raise ValueError("Manifest missing required field: method")
    if data["method"] not in ("dpo", "sft", "prompt"):
        raise ValueError(f"Invalid method: {data['method']}. Must be 'dpo', 'sft', or 'prompt'")
    if not data.get("data"):
        raise ValueError("Manifest missing required field: data")


def _build_output_result(output_data: dict[str, Any]) -> RunResult:
    """Build RunResult from output dict, parsing datetimes and nested adapters."""
    # Make a copy to avoid mutating the caller's dict
    data = dict(output_data)
    data["started_at"] = _parse_datetime(data.get("started_at"))
    data["completed_at"] = _parse_datetime(data.get("completed_at"))
    # Explicitly set adapter/parent to Adapter or None (not implicit empty state)
    data["adapter"] = Adapter(data["adapter"]) if data.get("adapter") else None
    data["parent"] = Adapter(data["parent"]) if data.get("parent") else None
    return RunResult(data)


def _build_manifest(data: dict[str, Any], source_path: Path | None = None) -> Manifest:
    """Build Manifest from validated dict.

    Args:
        data: Raw manifest data from YAML.
        source_path: Path to manifest file (for resolving relative external data paths).
    """
    _validate_required_fields(data)

    # Training section may contain output (new format) or output at root (old format)
    training_data = dict(data.get("training", {}))
    output_data = training_data.pop("output", None) or data.get("output")

    # Migrate old model section to training (backwards compatibility)
    if "model" in data:
        model_data = data["model"]
        if "requested_model" not in training_data and model_data.get("base"):
            training_data["requested_model"] = model_data["base"]

    output = _build_output_result(output_data) if output_data else None
    parent = Adapter(data["parent"]) if data.get("parent") else None

    return Manifest(
        version=data.get("version", 1),
        created_at=_parse_datetime(data.get("created_at")),
        method=data["method"],
        adapter=data["adapter"],
        source=Source(data.get("source", {})),
        parent=parent,
        lora=DotDict(data.get("lora", {})),
        training=DotDict(training_data),
        method_config=DotDict(data.get(data["method"], {})),
        data=Data(data["data"]),
        deployment=Deployment(data.get("deployment", {})),
        output=output,
        source_path=source_path,
    )


def load_manifest(path: Path) -> Manifest:
    """Load training manifest from YAML file (supports .yaml and .yaml.gz)."""
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    if path.stat().st_size == 0:
        raise CorruptedManifestError(path, "file is empty")

    data = _read_yaml_file(path)
    if not isinstance(data, dict):
        raise CorruptedManifestError(path, f"expected dict, got {type(data).__name__}")

    return _build_manifest(data, source_path=path.resolve())


def load_manifest_metadata(path: Path) -> Manifest:
    """Load manifest metadata only, skipping training data records.

    Much faster than load_manifest() for large files - streams gzip and stops
    before decompressing the data section. Use when you only need adapter info,
    parent lineage, or output metadata.

    The returned Manifest has data=Data() (empty) and should not be used for training.
    """
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    if path.stat().st_size == 0:
        raise CorruptedManifestError(path, "file is empty")

    data = _read_yaml_metadata(path)
    if not isinstance(data, dict):
        raise CorruptedManifestError(path, f"expected dict, got {type(data).__name__}")

    # Build manifest with empty data (metadata-only)
    data["data"] = {"format": "external", "path": "(metadata-only)"}
    return _build_manifest(data, source_path=path.resolve())


def _serialize_datetime(dt: datetime) -> str:
    """Serialize datetime to ISO format string."""
    return dt.isoformat()


def _get_effective_config(manifest: Manifest) -> tuple[dict, dict]:
    """Get effective lora/training config (from output if completed, else from manifest)."""
    output = manifest.get("output")
    manifest_lora = manifest.get("lora") or {}
    manifest_training = manifest.get("training") or {}

    if output is not None and output.get("config"):
        output_config = output.config
        lora_raw = output_config.get("lora", manifest_lora)
        training_raw = output_config.get("training", manifest_training)
        # Convert DotDicts to plain dicts for YAML serialization
        lora = lora_raw.to_dict() if hasattr(lora_raw, "to_dict") else dict(lora_raw)
        training = (
            training_raw.to_dict() if hasattr(training_raw, "to_dict") else dict(training_raw)
        )
        return lora, training
    return dict(manifest_lora), dict(manifest_training)


def _build_output_dict(output: RunResult) -> dict[str, Any]:
    """Build output dict for YAML serialization."""
    result: dict[str, Any] = {
        "status": output.status,
        "adapter": output.adapter.to_dict() if output.adapter else None,
        "base_model": output.base_model,
        "method": output.method,
        "metrics": output.metrics.to_dict() if output.metrics else None,
        "config": output.config.to_dict() if output.config else None,
        "started_at": _serialize_datetime(output.started_at),
        "completed_at": _serialize_datetime(output.completed_at),
        "samples_trained": output.samples_trained,
    }
    if output.parent is not None:
        result["parent"] = output.parent.to_dict()
    if output.error is not None:
        result["error"] = output.error
    return result


def _build_data_section(manifest: Manifest) -> dict[str, Any]:
    """Build data section dict for YAML serialization."""
    manifest_data = manifest.get("data") or Data()
    data_format = manifest_data.get("format", "inline")
    data_dict: dict[str, Any] = {"format": data_format}
    if data_format == "inline":
        data_dict["records"] = manifest_data.get("records", [])
    else:
        data_dict["path"] = manifest_data.get("path")
    return data_dict


def _build_training_section(manifest: Manifest) -> dict[str, Any]:
    """Build training section dict for YAML serialization (config only)."""
    _, training_config = _get_effective_config(manifest)
    return dict(training_config)


def _add_provenance_fields(data: dict[str, Any], manifest: Manifest) -> None:
    """Add source and parent fields to data dict if present."""
    source = manifest.get("source")
    if source and (
        source.get("context_key") or source.get("schema_name") or source.get("description")
    ):
        data["source"] = dict(source)
    parent = manifest.get("parent")
    if parent is not None:
        data["parent"] = dict(parent)


def _add_method_and_deployment(data: dict[str, Any], manifest: Manifest) -> None:
    """Add method-specific config and deployment fields to data dict if present."""
    method_config = manifest.get("method_config")
    if method_config:
        data[manifest.get("method")] = dict(method_config)
    deployment = manifest.get("deployment")
    if deployment and deployment.get("policy"):
        data["deployment"] = dict(deployment)


def _build_manifest_dict(manifest: Manifest) -> dict[str, Any]:
    """Build dict from Manifest for YAML serialization."""
    created_at = manifest.get("created_at") or datetime.now(UTC)
    data: dict[str, Any] = {
        "version": manifest.get("version", 1),
        "created_at": _serialize_datetime(created_at),
        "method": manifest.get("method"),
        "adapter": manifest.get("adapter"),
    }
    _add_provenance_fields(data, manifest)

    lora_config, _ = _get_effective_config(manifest)
    data["lora"] = lora_config
    data["training"] = _build_training_section(manifest)
    _add_method_and_deployment(data, manifest)

    output = manifest.get("output")
    if output is not None:
        data["output"] = _build_output_dict(output)
    data["data"] = _build_data_section(manifest)
    return data


class _FlowStyleDumper(yaml.SafeDumper):
    """YAML dumper that uses flow style for data records (compact, one per line)."""

    pass


class _QuotedStr(str):
    """String that serializes with double quotes in YAML."""

    pass


class _FlowDict(dict):
    """Dict that serializes in YAML flow style with quoted strings."""

    pass


def _quoted_str(dumper: yaml.SafeDumper, data: str) -> yaml.Node:
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style='"')


def _flow_style_dict(dumper: yaml.SafeDumper, data: dict) -> yaml.Node:
    return dumper.represent_mapping("tag:yaml.org,2002:map", data.items(), flow_style=True)


_FlowStyleDumper.add_representer(_FlowDict, _flow_style_dict)
_FlowStyleDumper.add_representer(_QuotedStr, _quoted_str)


def _record_to_flow(record: dict) -> _FlowDict:
    """Convert record to FlowDict with quoted keys and string values."""
    return _FlowDict(
        {_QuotedStr(k): _QuotedStr(v) if isinstance(v, str) else v for k, v in record.items()}
    )


def save_manifest(manifest: Manifest, path: Path, *, compress: bool = False) -> None:
    """Save training manifest to YAML file.

    Args:
        manifest: Manifest to save.
        path: Output path. If compress=True and path doesn't end in .gz, .gz is appended.
        compress: If True, gzip the output.
    """
    import gzip

    path.parent.mkdir(parents=True, exist_ok=True)
    data = _build_manifest_dict(manifest)

    # Use flow-style records for compact output
    if data["data"].get("format") == "inline":
        data["data"]["records"] = [_record_to_flow(r) for r in data["data"]["records"]]

    yaml_content = yaml.dump(
        data, Dumper=_FlowStyleDumper, sort_keys=False, allow_unicode=True, width=10000
    )

    if compress:
        if not path.suffix == ".gz":
            path = path.with_suffix(path.suffix + ".gz")
        with gzip.open(path, "wt", encoding="utf-8") as f:
            f.write(yaml_content)
    else:
        with path.open("w", encoding="utf-8") as f:
            f.write(yaml_content)


def _to_number(value: Any, default: int | float) -> int | float:
    """Convert value to number, handling string scientific notation."""
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default


def _validate_hyperparams(training: dict, lora: dict) -> list[str]:
    """Validate training hyperparameters.

    Note: Missing/None values pass validation (they use training defaults).
    Only explicitly invalid values (zero, negative) are flagged.
    """
    errors: list[str] = []
    if _to_number(training.get("num_epochs"), 3) <= 0:
        errors.append("num_epochs must be positive")
    if _to_number(training.get("batch_size"), 4) <= 0:
        errors.append("batch_size must be positive")
    if _to_number(training.get("learning_rate"), 2e-4) <= 0:
        errors.append("learning_rate must be positive")
    if _to_number(lora.get("r"), 16) <= 0:
        errors.append("LoRA rank must be positive")
    return errors


_VALID_DEPLOYMENT_POLICIES = frozenset(["skip", "add", "replace"])


def _validate_prompt_config(method_config: dict) -> list[str]:
    """Validate prompt tuning specific configuration."""
    errors: list[str] = []
    num_virtual_tokens = method_config.get("num_virtual_tokens", 20)
    prompt_tuning_init = method_config.get("prompt_tuning_init", "TEXT")
    prompt_tuning_init_text = method_config.get("prompt_tuning_init_text", "")
    if num_virtual_tokens <= 0:
        errors.append(f"num_virtual_tokens must be positive, got {num_virtual_tokens}")
    if prompt_tuning_init not in ("TEXT", "RANDOM"):
        errors.append(f"prompt_tuning_init must be 'TEXT' or 'RANDOM', got {prompt_tuning_init}")
    if prompt_tuning_init == "TEXT" and not prompt_tuning_init_text:
        errors.append("prompt_tuning_init_text is required when prompt_tuning_init='TEXT'")
    return errors


def _validate_core_fields(manifest: Manifest) -> list[str]:
    """Validate required manifest fields: adapter, method, deployment, data."""
    errors: list[str] = []
    adapter = manifest.get("adapter", "")
    method = manifest.get("method", "")
    data = manifest.get("data") or Data()

    if not adapter:
        errors.append("adapter is required")
    if method not in ("dpo", "sft", "prompt"):
        errors.append(f"method must be 'dpo', 'sft', or 'prompt', got '{method}'")
    if adapter and ("/" in adapter or "\\" in adapter or ".." in adapter):
        errors.append(f"Invalid adapter key: {adapter}")

    deployment = manifest.get("deployment") or Deployment()
    policy = deployment.get("policy")
    if policy is not None and policy not in _VALID_DEPLOYMENT_POLICIES:
        errors.append(f"deployment.policy must be 'skip', 'add', or 'replace', got '{policy}'")

    if data.get("format") == "inline" and not data.get("records"):
        errors.append("Inline data requires non-empty records list")
    elif data.get("format") == "external" and not data.get("path"):
        errors.append("External data requires a path")

    return errors


def validate_manifest(manifest: Manifest) -> list[str]:
    """Validate a training manifest."""
    errors = _validate_core_fields(manifest)

    training = manifest.get("training") or {}
    lora = manifest.get("lora") or {}
    errors.extend(_validate_hyperparams(training, lora))

    if manifest.get("method") == "prompt":
        errors.extend(_validate_prompt_config(manifest.get("method_config") or {}))

    return errors


def resolve_data(manifest: Manifest, work_dir: Path, manifest_path: Path | None = None) -> Path:
    """Resolve manifest data to a JSONL file path.

    Args:
        manifest: Training manifest with data section.
        work_dir: Working directory for training (inline data written here).
        manifest_path: Path to manifest file (for resolving relative external data paths).

    Returns:
        Path to JSONL data file.
    """
    if manifest.data.format == "external":
        if manifest.data.path is None:
            raise ValueError("External data format requires a path")
        ext_path = Path(manifest.data.path)
        # Absolute paths used as-is; relative paths resolved against manifest location
        if ext_path.is_absolute():
            data_path = ext_path
        elif manifest_path:
            data_path = manifest_path.parent / ext_path
        else:
            # Fallback to work_dir if no manifest path (in-memory manifest)
            data_path = work_dir / ext_path
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        return data_path

    # Write inline data to work directory
    work_dir.mkdir(parents=True, exist_ok=True)
    data_path = work_dir / f"{manifest.adapter}-data.jsonl"

    with data_path.open("w", encoding="utf-8") as f:
        for record in manifest.data.records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return data_path
