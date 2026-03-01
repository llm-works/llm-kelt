"""Training profile loading and config merging.

Handles loading named training profiles from app config and merging
with CLI overrides. Provides model-size-aware defaults for LoRA.

Based on empirical findings:
- 14B with r=16 performed similarly to 32B with r=32
- 32B needed r=64 to outperform smaller models
- Learning rate scales inversely with model size
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from appinfra import DotDict

from .schema import TRAINING_DEFAULTS

if TYPE_CHECKING:
    from .lora.config import Config

# ---------------------------------------------------------------------------
# Model-size-aware LoRA profiles
# ---------------------------------------------------------------------------

ModelSizeProfile = Literal["small", "medium", "large", "xlarge"]

MODEL_SIZE_PROFILES: dict[ModelSizeProfile, dict[str, Any]] = {
    "small": {  # ≤14B (7B, 8B, 13B, 14B)
        "lora": {"r": 16, "lora_alpha": 32},
        "training": {"learning_rate": 0.0002},
    },
    "medium": {  # 15B-50B (32B, 34B)
        "lora": {"r": 64, "lora_alpha": 128},
        "training": {"learning_rate": 0.0001},
    },
    "large": {  # 51B-70B (70B)
        "lora": {"r": 128, "lora_alpha": 256},
        "training": {"learning_rate": 0.00005},
    },
    "xlarge": {  # >70B (72B, 405B) - more conservative to avoid gradient explosion
        "lora": {"r": 128, "lora_alpha": 256},
        "training": {"learning_rate": 0.00002},
    },
}


def _extract_size_from_name(model_name: str) -> float | None:
    """Extract model size in billions from model name (fallback)."""
    match = re.search(r"(\d+(?:\.\d+)?)[Bb](?![a-zA-Z])", model_name)
    return float(match.group(1)) if match else None


def get_model_size_b(model_path: str) -> float | None:
    """Get model size in billions of parameters.

    Tries llm-infer metadata first (accurate), falls back to name parsing.

    Args:
        model_path: Model path or HuggingFace ID.

    Returns:
        Model size in billions, or None if not detectable.
    """
    # Try llm-infer metadata (requires model files to exist)
    try:
        from llm_infer.models import get_model_metadata

        meta = get_model_metadata(path=model_path)
        if meta.num_params_b is not None:
            return meta.num_params_b
    except (ImportError, FileNotFoundError, ValueError, AttributeError):
        pass

    # Fallback to name parsing
    return _extract_size_from_name(model_path)


def get_size_profile_name(size_b: float) -> ModelSizeProfile:
    """Get profile name for a given model size."""
    if size_b <= 14:
        return "small"
    elif size_b <= 50:
        return "medium"
    elif size_b <= 70:
        return "large"
    return "xlarge"


class ProfileDetectionError(ValueError):
    """Raised when model size cannot be auto-detected and no profile override given."""

    pass


def get_model_size_profile(
    model_path: str,
    *,
    profile_override: ModelSizeProfile | None = None,
    require_detection: bool = False,
) -> tuple[ModelSizeProfile, dict[str, Any]]:
    """Get appropriate LoRA profile for a model.

    Args:
        model_path: Model path or HuggingFace ID.
        profile_override: Explicit profile to use (skips auto-detection).
        require_detection: If True, raise error when auto-detection fails.

    Returns:
        Tuple of (profile_name, profile_config).

    Raises:
        ProfileDetectionError: If require_detection=True and size cannot be detected.
    """
    if profile_override:
        return profile_override, MODEL_SIZE_PROFILES[profile_override]

    size_b = get_model_size_b(model_path)
    if size_b is None:
        if require_detection:
            raise ProfileDetectionError(
                f"Cannot detect model size for '{model_path}'. "
                "Use --profile small|medium|large to specify manually."
            )
        return "small", MODEL_SIZE_PROFILES["small"]

    profile_name = get_size_profile_name(size_b)
    return profile_name, MODEL_SIZE_PROFILES[profile_name]


def build_lora_config(
    lora: DotDict,
    base_model: str,
    training: DotDict,
) -> tuple[str, Config]:
    """Build LoRA config with model-aware defaults.

    Args:
        lora: LoRA settings from manifest (may be empty).
        base_model: Model path for size detection.
        training: Training settings (may contain lora_profile override).

    Returns:
        Tuple of (profile_name, Config instance).
    """
    from .lora.config import Config

    profile_override = training.get("lora_profile")
    profile_name, profile = get_model_size_profile(
        base_model, profile_override=profile_override, require_detection=True
    )
    profile_lora = profile.get("lora", {})
    static_defaults = Config()

    config = Config(
        r=lora.get("r", profile_lora.get("r", static_defaults.r)),
        lora_alpha=lora.get(
            "lora_alpha", profile_lora.get("lora_alpha", static_defaults.lora_alpha)
        ),
        lora_dropout=lora.get("lora_dropout", static_defaults.lora_dropout),
        target_modules=lora.get("target_modules", static_defaults.target_modules),
        bias=lora.get("bias", static_defaults.bias),
        task_type=lora.get("task_type", static_defaults.task_type),
    )

    return profile_name, config


# ---------------------------------------------------------------------------
# App config profile loading
# ---------------------------------------------------------------------------


def load_default_profile(config: DotDict, method: str) -> DotDict:
    """Load default training profile for a method (sft/dpo).

    Args:
        config: App config (DotDict with training.default_profiles section).
        method: Training method ('sft' or 'dpo').

    Returns:
        Profile as DotDict, or empty DotDict if not configured.
    """
    training_cfg = getattr(config, "training", None)
    if training_cfg is None:
        return DotDict()

    profiles = getattr(training_cfg, "default_profiles", None)
    if profiles is None:
        return DotDict()

    profile = getattr(profiles, method, None)
    if profile is None:
        return DotDict()

    return DotDict(dict(profile))


def build_training_config(
    profile: DotDict | None = None,
    overrides: dict[str, Any] | None = None,
) -> DotDict:
    """Build training config by merging defaults, profile, and overrides.

    Args:
        profile: Optional profile config.
        overrides: Optional CLI/API overrides.

    Returns:
        Merged config as DotDict.
    """
    config = DotDict({**TRAINING_DEFAULTS})

    if profile:
        # Map profile keys to training config keys (aliased keys)
        key_map = {
            "epochs": "num_epochs",
            "batch_size": "batch_size",
            "learning_rate": "learning_rate",
            "fp16": "fp16",
            "bf16": "bf16",
        }
        for profile_key, config_key in key_map.items():
            if profile_key in profile:
                config[config_key] = profile[profile_key]
        # Direct config keys take precedence over aliased keys above
        for key in TRAINING_DEFAULTS:
            if key in profile:
                config[key] = profile[key]

    if overrides:
        for key, value in overrides.items():
            if value is not None:
                config[key] = value

    return config


def get_registry_path(config: DotDict) -> Path:
    """Get adapter registry path from app config.

    Args:
        config: App config DotDict.

    Returns:
        Path to adapter registry.

    Raises:
        ValueError: If not configured.
    """
    adapters_cfg = getattr(config, "adapters", None)
    lora_cfg = getattr(adapters_cfg, "lora", None) if adapters_cfg else None
    if not lora_cfg or not hasattr(lora_cfg, "base_path"):
        raise ValueError("adapters.lora.base_path not configured")
    return Path(lora_cfg.base_path)
