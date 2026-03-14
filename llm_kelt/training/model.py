"""Model utilities for training.

Provides model detection, quantization config, and training config helpers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from appinfra import DotDict
from llm_infer.models import get_model_metadata

from .schema import TRAINING_DEFAULTS

if TYPE_CHECKING:
    from appinfra.log import Logger


def is_model_quantized(base_model: str) -> bool:
    """Check if model is already quantized (GPTQ, AWQ, BNB).

    Args:
        base_model: Path or HuggingFace ID of model.

    Returns:
        True if model has pre-existing quantization.
    """
    try:
        meta = get_model_metadata(path=base_model)
        return meta.quantization is not None
    except (FileNotFoundError, ValueError):
        return False


def _load_stored_quantization_config(base_model: str) -> Any | None:
    """Load BitsAndBytesConfig from a pre-quantized model's config.json.

    Transformers has a bug where loading pre-quantized models without explicitly
    passing the quantization_config causes 'NoneType has no attribute to_dict'.
    This reconstructs the config from the stored JSON.
    """
    import json
    from pathlib import Path

    from transformers import BitsAndBytesConfig

    config_path = Path(base_model) / "config.json"
    if not config_path.exists():
        return None

    model_config = json.loads(config_path.read_text())
    quant_dict = model_config.get("quantization_config")
    if not quant_dict or quant_dict.get("quant_method") != "bitsandbytes":
        return None

    # Filter out private keys (start with _) that aren't BitsAndBytesConfig params
    filtered = {k: v for k, v in quant_dict.items() if not k.startswith("_")}
    return BitsAndBytesConfig(**filtered)


def get_quantization_config(
    lg: Logger,
    base_model: str,
    quantize_override: bool | None = None,
) -> tuple[Any, bool]:
    """Get BitsAndBytes quantization config based on model and override.

    Auto-detects: quantizes full-precision models, loads stored config if pre-quantized.

    Args:
        lg: Logger instance.
        base_model: Path or HuggingFace ID of model.
        quantize_override: Force quantization on/off. None = auto-detect.

    Returns:
        Tuple of (BitsAndBytesConfig or None, whether NEW quantization was applied).
        For pre-quantized models, returns (stored_config, False).
    """
    if quantize_override is not None:
        apply_quantization = quantize_override
    else:
        # Auto: quantize full-precision models, load stored config if already quantized
        is_prequantized = is_model_quantized(base_model)
        if is_prequantized:
            lg.info("model already quantized, loading stored config")
            stored_config = _load_stored_quantization_config(base_model)
            return stored_config, False  # Config loaded, but no new quantization applied
        apply_quantization = True

    if not apply_quantization:
        return None, False

    import torch
    from transformers import BitsAndBytesConfig

    config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    return config, True


# Keys that should be integers
_INT_KEYS = {
    "num_epochs",
    "batch_size",
    "gradient_accumulation_steps",
    "max_seq_length",
    "logging_steps",
    "save_steps",
    "seed",
}
# Keys that should be floats
_FLOAT_KEYS = {"learning_rate", "warmup_ratio", "eval_split"}


def _coerce_types(config: dict[str, Any]) -> dict[str, Any]:
    """Coerce config values to proper types (handles string numbers from YAML)."""
    result = dict(config)
    for key in _INT_KEYS:
        if key in result and result[key] is not None:
            try:
                result[key] = int(float(result[key]))
            except (ValueError, TypeError):
                pass
    for key in _FLOAT_KEYS:
        if key in result and result[key] is not None:
            try:
                result[key] = float(result[key])
            except (ValueError, TypeError):
                pass
    return result


def _apply_quantized_model_settings(
    lg: Logger, base_model: str, config: dict, overrides: dict
) -> None:
    """Disable AMP for pre-quantized models (modifies config in-place)."""
    try:
        meta = get_model_metadata(path=base_model)
        if meta.quantization:
            if "fp16" not in overrides:
                config["fp16"] = False
            if "bf16" not in overrides:
                config["bf16"] = False
            lg.info(
                "detected pre-quantized model, disabled AMP",
                extra={"quantization": meta.quantization, "bits": meta.quantization_bits},
            )
    except (FileNotFoundError, ValueError):
        pass  # Can't detect, use config values as-is


def build_training_config(
    lg: Logger,
    base_model: str,
    overrides: DotDict | dict | None = None,
) -> DotDict:
    """Build training config with model-aware defaults."""
    from .profiles import get_model_size_profile

    overrides = dict(overrides) if overrides else {}
    profile_override = overrides.pop("lora_profile", None)

    # Start with static defaults, apply size profile, then user overrides
    config = dict(TRAINING_DEFAULTS)
    profile_name, profile = get_model_size_profile(
        base_model, profile_override=profile_override, require_detection=True
    )
    config.update(profile.get("training", {}))
    config.update(overrides)
    config = _coerce_types(config)

    _apply_quantized_model_settings(lg, base_model, config, overrides)

    lg.info(
        f"training config (profile={profile_name})",
        extra={
            "learning_rate": config["learning_rate"],
            "num_epochs": config["num_epochs"],
            "batch_size": config["batch_size"],
            "max_grad_norm": config["max_grad_norm"],
            "max_seq_length": config["max_seq_length"],
            "fp16": config["fp16"],
            "bf16": config["bf16"],
        },
    )

    return DotDict(config)
