"""SFT (Supervised Fine-Tuning) training package.

Provides:
- Client: Training run management with example assignment
- export_run_examples: Export pending examples for a run to SFT format
"""

from .client import (
    Client,
    PendingExample,
    TrainedExample,
)
from .export import export_run_examples

__all__ = [
    # Core client
    "Client",
    # Models
    "PendingExample",
    "TrainedExample",
    # Export
    "export_run_examples",
]
