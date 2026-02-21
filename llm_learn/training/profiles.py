"""Training profile loading and config merging.

Handles loading named training profiles from app config and merging
with CLI overrides.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from appinfra import DotDict

from .schema import TRAINING_DEFAULTS


def load_profile(config: DotDict, profile_name: str) -> DotDict:
    """Load a named training profile from config.

    Args:
        config: App config (DotDict with training.profiles section).
        profile_name: Name of profile to load.

    Returns:
        Profile as DotDict.

    Raises:
        ValueError: If profile not found.
    """
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
        # Map profile keys to training config keys
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
        # Also accept direct config keys
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
