"""Training manifest module.

Provides file-based training workflow:
- Manifest: Self-contained training specification
- Client: Lifecycle management (create, load, save, submit)
- Runner: Execute training from manifest
"""

from .client import Client
from .runner import Result, Runner
from .schema import Data, Manifest, Model, Source

__all__ = [
    "Client",
    "Runner",
    "Result",
    "Manifest",
    "Source",
    "Model",
    "Data",
]
