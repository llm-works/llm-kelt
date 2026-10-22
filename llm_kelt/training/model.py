# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-kelt Authors

"""Model utilities for training.

Provides model detection, quantization config, and training config helpers.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

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


def _read_checkpoint_config(base_model: str) -> dict[str, Any] | None:
    """Read config.json from a local checkpoint, None when there is none."""
    config_path = Path(base_model) / "config.json"
    if not config_path.exists():
        return None
    return cast(dict[str, Any], json.loads(config_path.read_text()))


def _load_stored_quantization_config(base_model: str) -> Any | None:
    """Build the quantization config to load a pre-quantized model with for training.

    - bitsandbytes: reconstructs BitsAndBytesConfig from config.json. Transformers has a bug
      where loading pre-quantized models without explicitly passing the quantization_config
      causes 'NoneType has no attribute to_dict'.
    - compressed-tensors: CompressedTensorsConfig(dequantize=True). PEFT cannot attach LoRA
      to packed int4/int8 layers, so the weights are expanded to the model dtype at load.

    Reads the local checkpoint's config.json. Returns None when there is none (e.g. a hub ID)
    and for other formats (they load with the checkpoint's own config).
    """
    from transformers import BitsAndBytesConfig

    model_config = _read_checkpoint_config(base_model)
    if model_config is None:
        return None
    quant_dict = model_config.get("quantization_config") or {}
    quant_method = quant_dict.get("quant_method")
    if quant_method == "compressed-tensors":
        from transformers import CompressedTensorsConfig

        return CompressedTensorsConfig(dequantize=True)
    if quant_method != "bitsandbytes":
        return None

    # Filter out private keys (start with _) that aren't BitsAndBytesConfig params
    filtered = {k: v for k, v in quant_dict.items() if not k.startswith("_")}
    return BitsAndBytesConfig(**filtered)


def get_quantization_config(
    lg: Logger,
    base_model: str,
    quantize_override: bool | None = None,
) -> tuple[Any, bool]:
    """Get the quantization config to load a model for training with.

    A pre-quantized model always loads with its stored config (see
    _load_stored_quantization_config). For a full-precision model, quantize_override decides
    whether bitsandbytes 4-bit is applied; None = apply it.

    Args:
        lg: Logger instance.
        base_model: Path or HuggingFace ID of model.
        quantize_override: Force new quantization on/off. None = auto (on).

    Returns:
        Tuple of (quantization config or None, whether NEW quantization was applied).
        For pre-quantized models, returns (stored_config, False).
    """
    if is_model_quantized(base_model):
        if quantize_override:
            lg.info("model already quantized, ignoring quantize override")
        lg.info("model already quantized, loading stored config")
        return _load_stored_quantization_config(base_model), False

    if quantize_override is False:
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


# Class-name suffixes of language-model heads kelt can train. Any other class listed in a
# checkpoint's config (e.g. a classification head) falls back to AutoModelForCausalLM.
_TRAINABLE_ARCH_SUFFIXES = ("ForCausalLM", "ForConditionalGeneration")


def resolve_model_class(base_model: str) -> Any:
    """Return the transformers class a checkpoint was saved with, else AutoModelForCausalLM.

    Multimodal checkpoints (e.g. Qwen3.5 ``*ForConditionalGeneration``) name their language
    layers ``model.language_model.*``. Loaded through the text-only CausalLM class they become
    ``model.*``, so name-based settings stored in the checkpoint, such as a quantization
    ignore list, no longer match. Loading the saved class keeps the names, and the trained
    adapter's module names match what an inference server loads for the same checkpoint.
    """
    import transformers
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(base_model, trust_remote_code=True)
    for arch in getattr(config, "architectures", None) or []:
        cls = getattr(transformers, arch, None)
        if cls is not None and arch.endswith(_TRAINABLE_ARCH_SUFFIXES):
            return cls
    return AutoModelForCausalLM


def is_kbit_loaded(model: Any) -> bool:
    """True if bitsandbytes holds the model's weights in 4 or 8 bit (needs k-bit prep)."""
    return bool(
        getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False)
    )


def load_base_model(
    lg: Logger,
    base_model: str,
    quantize_override: bool | None,
    dtype: Any = None,
) -> Any:
    """Load a base model for training with auto-detected quantization.

    Args:
        lg: Logger instance.
        base_model: Path or HuggingFace ID of model.
        quantize_override: Force bitsandbytes quantization on/off. None = auto-detect.
        dtype: Torch dtype to load an unquantized model in; None keeps transformers'
            default. Ignored when a quantization config is used.

    Returns:
        The loaded model. Use is_kbit_loaded() to decide on k-bit training preparation.
    """
    quant_config, _ = get_quantization_config(lg, base_model, quantize_override)
    model_cls = resolve_model_class(base_model)
    # Only pass quantization_config if we're applying it - passing None overwrites
    # pre-quantized model configs and breaks transformers' to_dict() serialization
    model_kwargs: dict[str, Any] = {"device_map": "auto", "trust_remote_code": True}
    if quant_config is not None:
        model_kwargs["quantization_config"] = quant_config
    elif dtype is not None:
        model_kwargs["torch_dtype"] = dtype

    lg.info("loading model", extra={"model": base_model, "class": model_cls.__name__})
    return model_cls.from_pretrained(base_model, **model_kwargs)


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


def _checkpoint_dtype(base_model: str) -> str | None:
    """Weight dtype from a local checkpoint's config.json, None when there is none.

    Transformers writes it as `dtype`; older checkpoints carry `torch_dtype`.
    """
    model_config = _read_checkpoint_config(base_model)
    if model_config is None:
        return None
    dtype = model_config.get("dtype") or model_config.get("torch_dtype")
    return dtype if isinstance(dtype, str) else None


def _checkpoint_precision(lg: Logger, base_model: str) -> dict[str, bool] | None:
    """AMP precision for a local checkpoint, None when nothing calls for a change.

    - Pre-quantized models: AMP off.
    - bfloat16 weights: bf16 AMP. fp16 AMP's grad scaler cannot unscale bfloat16 gradients
      ("_amp_foreach_non_finite_check_and_unscale_cuda" not implemented for 'BFloat16').
    """
    try:
        meta = get_model_metadata(path=base_model)
    except (FileNotFoundError, ValueError):
        return None  # Can't detect, use config values as-is
    if meta.quantization:
        lg.info(
            "detected pre-quantized model, disabled AMP",
            extra={"quantization": meta.quantization, "bits": meta.quantization_bits},
        )
        return {"fp16": False, "bf16": False}
    if _checkpoint_dtype(base_model) == "bfloat16":
        lg.info("detected bfloat16 weights, using bf16 AMP")
        return {"fp16": False, "bf16": True}
    return None


def _apply_precision_settings(lg: Logger, base_model: str, config: dict, overrides: dict) -> None:
    """Settle fp16/bf16 in config (in-place); transformers rejects both enabled.

    An override that enables one of them picks the precision: the other is off unless the
    overrides set it too. Otherwise keys the overrides leave unset follow the checkpoint.
    """
    explicit = {key: bool(overrides[key]) for key in ("fp16", "bf16") if key in overrides}
    if any(explicit.values()):
        for key in ("fp16", "bf16"):
            config[key] = explicit.get(key, False)
        return
    for key, value in (_checkpoint_precision(lg, base_model) or {}).items():
        if key not in overrides:
            config[key] = value


def compute_warmup_steps(num_samples: int, tc: DotDict) -> int:
    """Convert ``tc.warmup_ratio`` into an absolute warmup step count.

    Trainers pass the result as ``warmup_steps``: transformers 5 removed the
    ``warmup_ratio`` argument, while an int step count means the same on every
    supported transformers version. Rounds up like ``TrainingArguments.get_warmup_steps``,
    so on a single device the count matches what HF derives from the ratio.
    """
    if tc.batch_size <= 0 or tc.gradient_accumulation_steps <= 0:
        raise ValueError("batch_size and gradient_accumulation_steps must be positive")

    ratio = float(tc.warmup_ratio)
    if not 0 <= ratio <= 1:
        raise ValueError("warmup_ratio must be in range [0, 1]")

    steps_per_epoch = max(
        1, math.ceil(num_samples / (tc.batch_size * tc.gradient_accumulation_steps))
    )
    total_steps: int = steps_per_epoch * tc.num_epochs
    return math.ceil(total_steps * ratio)


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

    _apply_precision_settings(lg, base_model, config, overrides)

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
