"""Training profile loading and config merging.

Handles loading named training profiles from app config and merging
with CLI overrides.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from appinfra import DotDict

from .schema import TRAINING_DEFAULTS


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
