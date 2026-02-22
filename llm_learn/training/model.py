"""Model utilities for training.

Provides model detection and training config helpers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from appinfra import DotDict
from llm_infer.models import get_model_metadata

from .schema import TRAINING_DEFAULTS

if TYPE_CHECKING:
    from appinfra.log import Logger

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
