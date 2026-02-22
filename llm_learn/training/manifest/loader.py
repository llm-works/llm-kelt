"""Manifest loading and saving.

Handles YAML serialization of training manifests and data resolution.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import yaml
from appinfra import DotDict

from ..schema import Adapter, RunResult
from .errors import CorruptedManifestError
from .schema import Data, Manifest, Source


def _parse_datetime(value: str | datetime) -> datetime:
    """Parse datetime from string or pass through."""
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(value)


def _serialize_datetime(dt: datetime) -> str:
    """Serialize datetime to ISO format string."""
    return dt.isoformat()


def _dict_to_source(data: dict[str, Any] | None) -> Source:
    """Convert dict to Source."""
    if data is None:
        return Source()
    return Source(context_key=data.get("context_key"), description=data.get("description"))


def _dict_to_data(data: dict[str, Any]) -> Data:
    """Convert dict to Data."""
    return Data(
        format=data.get("format", "inline"), records=data.get("records", []), path=data.get("path")
    )


def _dict_to_adapter(data: dict[str, Any] | None) -> Adapter | None:
    """Convert dict to Adapter."""
    if data is None:
        return None
    return Adapter(md5=data["md5"], mtime=data["mtime"], path=data["path"])


def _dict_to_output(data: dict[str, Any] | None) -> RunResult | None:
    """Convert dict to RunResult."""
    if data is None:
        return None
    return RunResult(
        status=data["status"],
        adapter=_dict_to_adapter(data["adapter"]) if data.get("adapter") else Adapter("", "", ""),
        base_model=data.get("base_model", ""),
        method=data.get("method", ""),
        metrics=data.get("metrics", {}),
        config=data.get("config", {}),
        started_at=_parse_datetime(data["started_at"])
        if data.get("started_at")
        else datetime.now(),
        completed_at=_parse_datetime(data["completed_at"])
        if data.get("completed_at")
        else datetime.now(),
        samples_trained=data.get("samples_trained", 0),
        parent=_dict_to_adapter(data.get("parent")),
        error=data.get("error"),
    )


def _validate_required_fields(
    data: dict[str, Any],
) -> tuple[str, Literal["dpo", "sft"], dict[str, Any]]:
    """Validate and extract required manifest fields."""
    adapter = data.get("adapter")
    method = data.get("method")
    data_section = data.get("data")

    if not adapter:
        raise ValueError("Manifest missing required field: adapter")
    if not method:
        raise ValueError("Manifest missing required field: method")
    if method not in ("dpo", "sft"):
        raise ValueError(f"Invalid method: {method}. Must be 'dpo' or 'sft'")
    if not data_section:
        raise ValueError("Manifest missing required field: data")

    return adapter, method, data_section  # type: ignore[return-value]


def _read_yaml_file(path: Path) -> dict:
    """Read YAML from file (supports .gz)."""
    import gzip

    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return dict(yaml.safe_load(f))
    with path.open("r", encoding="utf-8") as f:
        return dict(yaml.safe_load(f))


def _dict_to_manifest(data: dict) -> Manifest:
    """Convert validated dict to Manifest object."""
    adapter, method, data_section = _validate_required_fields(data)

    # Training section may contain output (new format) or output at root (old format)
    training_data = data.get("training", {})
    output_data = training_data.pop("output", None) or data.get("output")

    # Migrate old model section to training (backwards compatibility)
    if "model" in data:
        model_data = data["model"]
        if "requested_model" not in training_data and model_data.get("base"):
            training_data["requested_model"] = model_data["base"]

    return Manifest(
        version=data.get("version", 1),
        created_at=_parse_datetime(data.get("created_at", datetime.now().astimezone())),
        method=method,
        adapter=adapter,
        source=_dict_to_source(data.get("source")),
        parent=_dict_to_adapter(data.get("parent")),
        lora=DotDict(data.get("lora", {})),
        training=DotDict(training_data),
        method_config=DotDict(data.get(method, {})),
        data=_dict_to_data(data_section),
        output=_dict_to_output(output_data),
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

    return _dict_to_manifest(data)


def _source_to_dict(source: Source) -> dict[str, Any] | None:
    """Convert Source to dict, returning None if empty."""
    if source.context_key is None and source.description is None:
        return None
    result: dict[str, Any] = {}
    if source.context_key is not None:
        result["context_key"] = source.context_key
    if source.description is not None:
        result["description"] = source.description
    return result


def _data_to_dict(data: Data) -> dict[str, Any]:
    """Convert Data to dict."""
    result: dict[str, Any] = {"format": data.format}
    if data.format == "inline":
        result["records"] = data.records
    else:
        result["path"] = data.path
    return result


def _adapter_to_dict(adapter: Adapter | None) -> dict[str, Any] | None:
    """Convert Adapter to dict."""
    if adapter is None:
        return None
    return {"md5": adapter.md5, "mtime": adapter.mtime, "path": adapter.path}


def _output_to_dict(output: RunResult | None) -> dict[str, Any] | None:
    """Convert RunResult to dict for YAML serialization."""
    if output is None:
        return None
    result: dict[str, Any] = {
        "status": output.status,
        "adapter": _adapter_to_dict(output.adapter),
        "base_model": output.base_model,
        "method": output.method,
        "metrics": output.metrics,
        "config": output.config,
        "started_at": _serialize_datetime(output.started_at),
        "completed_at": _serialize_datetime(output.completed_at),
        "samples_trained": output.samples_trained,
    }
    if output.parent is not None:
        result["parent"] = _adapter_to_dict(output.parent)
    if output.error is not None:
        result["error"] = output.error
    return result


def _get_effective_config(manifest: Manifest) -> tuple[dict, dict]:
    """Get effective lora/training config (from output if completed, else from manifest)."""
    if manifest.output is not None and manifest.output.config:
        output_config = manifest.output.config
        lora = output_config.get("lora", dict(manifest.lora) if manifest.lora else {})
        training = output_config.get(
            "training", dict(manifest.training) if manifest.training else {}
        )
        return lora, training
    return (
        dict(manifest.lora) if manifest.lora else {},
        dict(manifest.training) if manifest.training else {},
    )


def _manifest_to_dict(manifest: Manifest) -> dict[str, Any]:
    """Convert Manifest to dict for YAML serialization.

    When output is present (completed), output is nested under training section.
    """
    data: dict[str, Any] = {
        "version": manifest.version,
        "created_at": _serialize_datetime(manifest.created_at),
        "method": manifest.method,
        "adapter": manifest.adapter,
    }

    if (source_dict := _source_to_dict(manifest.source)) is not None:
        data["source"] = source_dict
    if manifest.parent is not None:
        data["parent"] = _adapter_to_dict(manifest.parent)

    lora_config, training_config = _get_effective_config(manifest)
    data["lora"] = lora_config

    # Nest output under training section
    training_dict: dict[str, Any] = training_config
    if manifest.output is not None:
        training_dict["output"] = _output_to_dict(manifest.output)
    data["training"] = training_dict

    if manifest.method_config:
        data[manifest.method] = dict(manifest.method_config)
    data["data"] = _data_to_dict(manifest.data)

    return data


class _FlowStyleDumper(yaml.SafeDumper):
    """YAML dumper that uses flow style for data records (compact, one per line)."""

    pass


def _quoted_str(dumper: yaml.SafeDumper, data: str) -> yaml.Node:
    """Represent string with double quotes."""
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style='"')


class QuotedStr(str):
    """String that serializes with double quotes in YAML."""

    pass


class FlowDict(dict):
    """Dict that serializes in YAML flow style with quoted strings."""

    pass


def _flow_style_dict(dumper: yaml.SafeDumper, data: dict) -> yaml.Node:
    """Represent dict in flow style (JSON-like, single line)."""
    return dumper.represent_mapping("tag:yaml.org,2002:map", data.items(), flow_style=True)


_FlowStyleDumper.add_representer(FlowDict, _flow_style_dict)
_FlowStyleDumper.add_representer(QuotedStr, _quoted_str)


def _record_to_flow(record: dict) -> FlowDict:
    """Convert record to FlowDict with quoted keys and string values (JSON-like)."""
    return FlowDict(
        {QuotedStr(k): QuotedStr(v) if isinstance(v, str) else v for k, v in record.items()}
    )


def _data_to_dict_flow(data: Data) -> dict[str, Any]:
    """Convert Data to dict, using flow style for inline records."""
    result: dict[str, Any] = {"format": data.format}
    if data.format == "inline":
        result["records"] = [_record_to_flow(r) for r in data.records]
    else:
        result["path"] = data.path
    return result


def save_manifest(manifest: Manifest, path: Path, *, compress: bool = False) -> None:
    """Save training manifest to YAML file.

    Args:
        manifest: Manifest to save.
        path: Output path. If compress=True and path doesn't end in .gz, .gz is appended.
        compress: If True, gzip the output.
    """
    import gzip

    path.parent.mkdir(parents=True, exist_ok=True)
    data = _manifest_to_dict(manifest)
    # Use flow-style records for compact output
    if "data" in data and data["data"].get("format") == "inline":
        data["data"] = _data_to_dict_flow(manifest.data)

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


def _validate_data(manifest: Manifest) -> list[str]:
    """Validate manifest data section (structural only)."""
    errors: list[str] = []
    if manifest.data.format == "inline" and not manifest.data.records:
        errors.append("Inline data requires non-empty records list")
    elif manifest.data.format == "external" and not manifest.data.path:
        errors.append("External data requires a path")
    return errors


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


def validate_manifest(manifest: Manifest) -> list[str]:
    """Validate a training manifest."""
    errors: list[str] = []

    if not manifest.adapter:
        errors.append("adapter is required")
    if manifest.method not in ("dpo", "sft"):
        errors.append(f"method must be 'dpo' or 'sft', got '{manifest.method}'")

    errors.extend(_validate_data(manifest))

    if _to_number(manifest.training.get("num_epochs"), 3) <= 0:
        errors.append("num_epochs must be positive")
    if _to_number(manifest.training.get("batch_size"), 4) <= 0:
        errors.append("batch_size must be positive")
    if _to_number(manifest.training.get("learning_rate"), 2e-4) <= 0:
        errors.append("learning_rate must be positive")
    if _to_number(manifest.lora.get("r"), 16) <= 0:
        errors.append("LoRA rank must be positive")
    # target_modules can be empty - trainer applies defaults

    return errors


def resolve_data(manifest: Manifest, work_dir: Path) -> Path:
    """Resolve manifest data to a JSONL file path.

    Args:
        manifest: Training manifest with data section.
        work_dir: Working directory for training (inline data written here).

    Returns:
        Path to JSONL data file.
    """
    if manifest.data.format == "external":
        if manifest.data.path is None:
            raise ValueError("External data format requires a path")
        data_path = work_dir / manifest.data.path
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
