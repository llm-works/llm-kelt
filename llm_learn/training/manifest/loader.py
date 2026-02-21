"""Manifest loading and saving.

Handles YAML serialization of training manifests and data resolution.
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import yaml
from appinfra import DotDict

from .schema import (
    Data,
    Manifest,
    Model,
    Source,
)


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
    return Source(
        context_key=data.get("context_key"),
        description=data.get("description"),
    )


def _dict_to_model(data: dict[str, Any] | None) -> Model:
    """Convert dict to Model."""
    if data is None:
        return Model()
    return Model(
        base=data.get("base", "Qwen/Qwen2.5-7B-Instruct"),
        quantize=data.get("quantize", True),
    )


def _dict_to_data(data: dict[str, Any]) -> Data:
    """Convert dict to Data."""
    return Data(
        format=data.get("format", "inline"),
        records=data.get("records", []),
        path=data.get("path"),
    )


def _validate_required_fields(
    data: dict[str, Any],
) -> tuple[str, Literal["dpo", "sft"], dict[str, Any]]:
    """Validate and extract required manifest fields."""
    adapter_id = data.get("adapter_id")
    method = data.get("method")
    data_section = data.get("data")

    if not adapter_id:
        raise ValueError("Manifest missing required field: adapter_id")
    if not method:
        raise ValueError("Manifest missing required field: method")
    if method not in ("dpo", "sft"):
        raise ValueError(f"Invalid method: {method}. Must be 'dpo' or 'sft'")
    if not data_section:
        raise ValueError("Manifest missing required field: data")

    return adapter_id, method, data_section  # type: ignore[return-value]


def load_manifest(path: Path) -> Manifest:
    """Load training manifest from YAML file.

    Args:
        path: Path to manifest YAML file.

    Returns:
        Manifest instance.

    Raises:
        FileNotFoundError: If manifest file doesn't exist.
        ValueError: If manifest is invalid.
    """
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Invalid manifest format: expected dict, got {type(data)}")

    adapter_id, method, data_section = _validate_required_fields(data)

    # Method-specific config (e.g., "dpo" section for DPO manifests)
    method_config = data.get(method, {})

    return Manifest(
        version=data.get("version", 1),
        created_at=_parse_datetime(data.get("created_at", datetime.now().astimezone())),
        method=method,
        adapter_id=adapter_id,
        source=_dict_to_source(data.get("source")),
        model=_dict_to_model(data.get("model")),
        parent_adapter=data.get("parent_adapter"),
        lora=DotDict(data.get("lora", {})),
        training=DotDict(data.get("training", {})),
        method_config=DotDict(method_config),
        data=_dict_to_data(data_section),
    )


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


def _model_to_dict(model: Model) -> dict[str, Any]:
    """Convert Model to dict."""
    return {
        "base": model.base,
        "quantize": model.quantize,
    }


def _data_to_dict(data: Data) -> dict[str, Any]:
    """Convert Data to dict."""
    result: dict[str, Any] = {"format": data.format}
    if data.format == "inline":
        result["records"] = data.records
    else:
        result["path"] = data.path
    return result


def save_manifest(manifest: Manifest, path: Path) -> None:
    """Save training manifest to YAML file.

    Args:
        manifest: Manifest to save.
        path: Output path for YAML file.
    """
    data: dict[str, Any] = {
        "version": manifest.version,
        "created_at": _serialize_datetime(manifest.created_at),
        "method": manifest.method,
        "adapter_id": manifest.adapter_id,
        "model": _model_to_dict(manifest.model),
        "lora": dict(manifest.lora) if manifest.lora else {},
        "training": dict(manifest.training) if manifest.training else {},
        "data": _data_to_dict(manifest.data),
    }

    # Optional fields
    source_dict = _source_to_dict(manifest.source)
    if source_dict is not None:
        data["source"] = source_dict

    if manifest.parent_adapter is not None:
        data["parent_adapter"] = manifest.parent_adapter

    # Method-specific config saved under method name (e.g., "dpo": {...})
    if manifest.method_config:
        data[manifest.method] = dict(manifest.method_config)

    # Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def _validate_data(manifest: Manifest) -> list[str]:
    """Validate manifest data section (structural only).

    Method-specific record validation is done by dpo/sft clients.
    """
    errors: list[str] = []
    if manifest.data.format == "inline" and not manifest.data.records:
        errors.append("Inline data requires non-empty records list")
    elif manifest.data.format == "external" and not manifest.data.path:
        errors.append("External data requires a path")
    return errors


def validate_manifest(manifest: Manifest) -> list[str]:
    """Validate a training manifest.

    Args:
        manifest: Manifest to validate.

    Returns:
        List of validation error messages (empty if valid).
    """
    errors: list[str] = []

    # Required fields
    if not manifest.adapter_id:
        errors.append("adapter_id is required")
    if manifest.method not in ("dpo", "sft"):
        errors.append(f"method must be 'dpo' or 'sft', got '{manifest.method}'")

    # Data, training, and LoRA validation
    errors.extend(_validate_data(manifest))

    if manifest.training.get("num_epochs", 3) <= 0:
        errors.append("num_epochs must be positive")
    if manifest.training.get("batch_size", 4) <= 0:
        errors.append("batch_size must be positive")
    if manifest.training.get("learning_rate", 2e-4) <= 0:
        errors.append("learning_rate must be positive")
    if manifest.lora.get("r", 16) <= 0:
        errors.append("LoRA rank must be positive")
    if not manifest.lora.get("target_modules"):
        errors.append("target_modules cannot be empty")

    return errors


def resolve_data(manifest: Manifest, manifest_dir: Path) -> Path:
    """Resolve manifest data to a JSONL file path.

    For inline data, writes records to a temporary file.
    For external data, resolves the path relative to manifest directory.

    Args:
        manifest: Manifest with data to resolve.
        manifest_dir: Directory containing the manifest (for relative paths).

    Returns:
        Path to JSONL file containing training data.

    Raises:
        FileNotFoundError: If external data file doesn't exist.
    """
    if manifest.data.format == "external":
        if manifest.data.path is None:
            raise ValueError("External data format requires a path")
        data_path = manifest_dir / manifest.data.path
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        return data_path

    # Inline format - write to temp file
    # Use a named temp file that won't be auto-deleted
    temp_dir = Path(tempfile.gettempdir()) / "llm-learn-manifests"
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Use adapter_id in filename for easier debugging
    temp_path = temp_dir / f"{manifest.adapter_id}-data.jsonl"

    with temp_path.open("w", encoding="utf-8") as f:
        for record in manifest.data.records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return temp_path
