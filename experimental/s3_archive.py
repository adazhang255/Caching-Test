"""S3/Glacier TTL archive adapter (external to LMCache).

This module owns TTL. LMCache never writes to S3.

In dev (no AWS), falls back to local filesystem root.
"""
from __future__ import annotations

import os
import time
import json
from typing import Optional, Dict, Any

try:
    import boto3  # type: ignore
except Exception:
    boto3 = None

from .engine_serialization import compress_bytes, decompress_bytes


class S3Archive:
    def __init__(self, bucket: Optional[str] = None, prefix: str = "lmcache-archive/"):
        self.bucket = bucket or os.getenv("LMCACHE_S3_BUCKET")
        self.prefix = prefix.strip("/") + "/"
        self.local_root = os.getenv("LMCACHE_ARCHIVE_LOCAL", os.path.join(os.getcwd(), "archive_store"))
        if self.bucket and boto3 is None:
            raise RuntimeError("boto3 not available but S3 bucket configured")
        if not self.bucket:
            os.makedirs(self.local_root, exist_ok=True)

    def _s3_key(self, key: str) -> str:
        return f"{self.prefix}{key}.gz"

    def put_kv(self, key: str, blob: bytes, ttl_seconds: int, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        meta = meta or {}
        payload = compress_bytes(blob)
        expires_at = int(time.time()) + int(ttl_seconds)

        if self.bucket:
            s3 = boto3.client("s3")  # type: ignore
            s3.put_object(
                Bucket=self.bucket,
                Key=self._s3_key(key),
                Body=payload,
                Metadata={"expires_at": str(expires_at), **{f"m_{k}": str(v) for k, v in meta.items()}},
                StorageClass=os.getenv("LMCACHE_S3_STORAGE_CLASS", "STANDARD"),
            )
        else:
            path = os.path.join(self.local_root, f"{key}.gz")
            with open(path, "wb") as f:
                f.write(payload)
            idx = os.path.join(self.local_root, f"{key}.json")
            with open(idx, "w") as f:
                json.dump({"expires_at": expires_at, "meta": meta}, f)
        return {"key": key, "expires_at": expires_at}

    def get_kv_if_fresh(self, key: str) -> Optional[Dict[str, Any]]:
        now = int(time.time())
        if self.bucket:
            s3 = boto3.client("s3")  # type: ignore
            try:
                head = s3.head_object(Bucket=self.bucket, Key=self._s3_key(key))
            except Exception:
                return None
            meta = head.get("Metadata", {})
            exp = int(meta.get("expires_at", "0"))
            if now >= exp:
                return None
            obj = s3.get_object(Bucket=self.bucket, Key=self._s3_key(key))
            data = obj["Body"].read()
            blob = decompress_bytes(data)
            return {"key": key, "blob": blob, "expires_at": exp, "meta": meta}
        else:
            path = os.path.join(self.local_root, f"{key}.gz")
            idx = os.path.join(self.local_root, f"{key}.json")
            if not (os.path.exists(path) and os.path.exists(idx)):
                return None
            try:
                with open(idx, "r") as f:
                    info = json.load(f)
            except Exception:
                return None
            if now >= int(info.get("expires_at", 0)):
                return None
            with open(path, "rb") as f:
                blob = decompress_bytes(f.read())
            return {"key": key, "blob": blob, "expires_at": info["expires_at"], "meta": info.get("meta", {})}

