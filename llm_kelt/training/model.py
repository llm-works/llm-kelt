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


def get_quantization_config(
    lg: Logger,
    base_model: str,
    quantize_override: bool | None = None,
) -> tuple[Any, bool]:
    """Get BitsAndBytes quantization config based on model and override.

    Auto-detects: quantizes full-precision models, skips if already quantized.

    Args:
        lg: Logger instance.
        base_model: Path or HuggingFace ID of model.
        quantize_override: Force quantization on/off. None = auto-detect.

    Returns:
        Tuple of (BitsAndBytesConfig or None, whether quantization was applied).
    """
    if quantize_override is not None:
        apply_quantization = quantize_override
    else:
        # Auto: quantize full-precision models, skip if already quantized
        is_prequantized = is_model_quantized(base_model)
        if is_prequantized:
            lg.info("model already quantized, skipping BNB quantization")
        apply_quantization = not is_prequantized

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


def build_training_config(
    lg: Logger,
    base_model: str,
    overrides: DotDict | dict | None = None,
) -> DotDict:
    """Build training config with model-aware defaults.

    Detects pre-quantized models (BNB, GPTQ, AWQ) and disables fp16/bf16
    to avoid AMP conflicts.

    Args:
        lg: Logger instance.
        base_model: Path or HuggingFace ID of base model.
        overrides: User-provided config overrides.

    Returns:
        DotDict with merged training config.
    """
    config = dict(TRAINING_DEFAULTS)
    overrides = dict(overrides) if overrides else {}
    config.update(overrides)

    # Coerce string values to proper types
    config = _coerce_types(config)

    # Check if model is pre-quantized
    try:
        meta = get_model_metadata(path=base_model)
        if meta.quantization:
            # Pre-quantized models don't work with AMP
            if "fp16" not in overrides:
                config["fp16"] = False
            if "bf16" not in overrides:
                config["bf16"] = False
            lg.info(
                "detected pre-quantized model, disabled AMP",
                extra={"quantization": meta.quantization, "bits": meta.quantization_bits},
            )
    except (FileNotFoundError, ValueError):
        # Can't detect, use config values as-is
        pass

    return DotDict(config)
