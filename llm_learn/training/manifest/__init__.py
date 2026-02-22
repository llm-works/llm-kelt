"""Training manifest module.

Provides file-based training workflow:
- Manifest: Self-contained training specification
- Client: Lifecycle management (create, load, save, submit)
"""

from .client import Client
from .errors import CorruptedManifestError, ManifestError
from .schema import Data, Manifest, Source

__all__ = [
    "Client",
    "CorruptedManifestError",
    "Data",
    "Manifest",
    "ManifestError",
    "Source",
]
