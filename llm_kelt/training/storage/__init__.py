"""Storage abstraction for training artifacts.

Provides abstract Storage interface and FileStorage implementation.
"""

from .base import DupAdapterError, Storage, extract_md5, md5_matches
from .file import FileStorage

__all__ = ["DupAdapterError", "Storage", "FileStorage", "extract_md5", "md5_matches"]
