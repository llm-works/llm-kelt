"""SFT (Supervised Fine-Tuning) training package.

Provides:
- Client: Training client that operates on manifests

Note: Actual SFT training is provided by llm_learn.training.lora.train_lora.
"""

from .client import Client

__all__ = [
    "Client",
]
