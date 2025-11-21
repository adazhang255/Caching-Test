# src/__init__.py
"""LMCache TTL-Heuristics Utilities.

Client + heuristics for working with an external LMCache controller.
"""

from .cache_controller import LMCacheController
from .caching_heuristics import compute_ttl, select_backend
from .tiered_caching import (
    MultiTierCache,
)

__version__ = "0.1.0"
__author__ = "LMCache Contributors"
__all__ = [
    "LMCacheController",
    "compute_ttl",
    "select_backend",
    "BaseBackend",
    "InMemoryBackend",
    "LocalDiskBackend",
    "RemoteBackend",
    "MultiTierCache",
]
