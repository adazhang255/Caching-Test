"""Background worker to archive LMCache KV to S3 with TTL.

Flow:
  - Enumerate LMCache entries (GPU/Disk) via controller or local index
  - Serialize + compress
  - s3_archive.put_kv(key, blob, ttl)
  - Never mutate LMCache; this is write-only to S3

Note: Controller enumeration and KV serialization are environment-specific.
Provide hooks to integrate with your LMCache internals.
"""
from __future__ import annotations

import os
import time
from typing import Iterable, Tuple

from .s3_archive import S3Archive


def iter_kv_entries() -> Iterable[Tuple[str, bytes, int]]:
    """Yield (key, serialized_blob, ttl_seconds) for LMCache entries.

    TODO: Implement actual enumeration & serialization from LMCache GPU/Disk.
    This is a stub returning empty iterator.
    """
    return []


def run_once() -> int:
    arch = S3Archive()
    count = 0
    for key, blob, ttl in iter_kv_entries():
        arch.put_kv(key, blob, ttl)
        count += 1
    return count


def main_loop(interval_sec: int = 60):
    while True:
        try:
            n = run_once()
            print(f"Archived {n} entries")
        except Exception as e:
            print(f"archive worker error: {e}")
        time.sleep(interval_sec)


if __name__ == "__main__":
    main_loop()

