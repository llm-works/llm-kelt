"""Storage abstraction for training artifacts.

Provides abstract Storage interface and FileStorage implementation.
"""

from .base import Storage
from .file import FileStorage

__all__ = ["Storage", "FileStorage"]
