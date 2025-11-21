"""TTL and backend selection heuristics for LMCache GPU+Disk only.

This module exposes compute_ttl(...) and select_backend(...) functions which
map prompt metadata (perplexity, time_variance, access_count, size, cost)
into a TTL (seconds) and a suggested storage backend name. The functions are
kept small and configurable so you can iterate on formulas quickly.
"""

from typing import Dict, Any
import os
import math


def clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def compute_ttl(metadata: Dict[str, Any],
                base_ttl: int = 3600,
                min_ttl: int = 0,
                max_ttl: int = 7 * 24 * 3600,
                alpha: float = 0.25):
    """Compute TTL in seconds based on metadata.

    metadata keys used (optional):
      - perplexity: float (higher -> likely more unique -> longer TTL?)
      - access_count: int (more accesses -> increase TTL)
      - time_variance: float in [0..1] (higher variance -> reduce TTL)

    Formula (starter):
      ttl = base_ttl * (1 + alpha * log(1+access_count)) * (1 + beta * cost_factor) * (1 - gamma*time_variance)

    The function clamps results to [min_ttl, max_ttl].
    """

    perplexity = float(metadata.get("perplexity", 1.0))
    access_count = int(metadata.get("access_count", 0))
    time_variance = float(metadata.get("time_variance", 0.0))

    # access influence (log-smooth)
    access_influence = 1.0 + alpha * math.log1p(access_count)

    # perplexity influence: by default, higher perplexity indicates more surprising
    # outputs — we may want to keep them longer (multiply factor). scale modestly.
    perplexity_factor = 1.0 + (max(0.0, perplexity - 10.0) / 100.0)

    # time_variance penalizes TTL (if high volatility, lower TTL)
    gamma = 0.8

    ttl = base_ttl * access_influence * perplexity_factor * (1.0 - gamma * clamp(time_variance, 0.0, 1.0))

    ttl = int(clamp(ttl, min_ttl, max_ttl))
    return ttl


def select_backend(metadata: Dict[str, Any],
                   hot_threshold_perplexity: float = 20.0,
                   recent_seconds: int = 3600) -> str:
    """Decide preferred backend for entries.

    Only returns "gpu" (hot) or "disk" (warm). Remote is disabled by design.
    """

    # Optional guard: prevent offloading beyond GPU if set.
    if os.getenv("LMCACHE_DISABLE_OFFLOAD", "0") == "1":
        return "gpu"

    perplexity = float(metadata.get("perplexity", 0.0))
    access_count = int(metadata.get("access_count", 0))
    last_accessed = float(metadata.get("last_accessed", 0.0))
    time_variance = float(metadata.get("time_variance", 0.0))

    now = metadata.get("now", None)
    # simple recency check if 'now' provided
    recency_score = 0
    if now and last_accessed:
        recency_score = max(0, now - last_accessed)

    # Heuristic rules (GPU+Disk only):
    # - Hot & recently accessed → gpu
    # - Volatile or less recent → disk

    # if perplexity <= hot_threshold_perplexity and access_count >= 2 and (now is None or recency_score <= recent_seconds):
    #     return "gpu"
    # return "disk"
    
    if compute_ttl(metadata) < 10:
        return "gpu"
    return "disk"


__all__ = ["compute_ttl", "select_backend"]
