"""Storage abstraction for training artifacts.

Provides abstract Storage interface and FileStorage implementation.
"""

from .base import DupAdapterError, Storage
from .file import FileStorage

__all__ = ["DupAdapterError", "Storage", "FileStorage"]
