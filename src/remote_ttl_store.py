import os
import time
import json
import hashlib
from typing import Optional, Dict, Any

try:
    import fakeredis  # type: ignore
except Exception:
    fakeredis = None


def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


class RemoteTTLStore:
    """Simple TTL-backed remote store using fakeredis for index + filesystem blobs.

    This is a development adapter. Swap filesystem blobs with S3 later by
    replacing _write_blob/_read_blob with boto3 put_object/get_object.
    """

    def __init__(self, root_dir: Optional[str] = None, namespace: str = "lmcache:ttl"):
        if fakeredis is None:
            raise RuntimeError("fakeredis is required for RemoteTTLStore in dev.")
        self.r = fakeredis.FakeStrictRedis()
        self.ns = namespace.rstrip(":")
        self.root = root_dir or os.getenv("REMOTE_BLOB_ROOT", os.path.join(os.getcwd(), "remote_blobs"))
        os.makedirs(self.root, exist_ok=True)

    def _idx_key(self, key: str) -> str:
        return f"{self.ns}:idx:{key}"

    def _blob_path(self, key: str) -> str:
        return os.path.join(self.root, key)

    def put(self, logical_key: str, payload: str | bytes, ttl_seconds: int, content_type: str = "text/plain") -> Dict[str, Any]:
        key = _sha256(logical_key)
        blob_path = self._blob_path(key)

        self._write_blob(blob_path, payload)
        meta = {"content_type": content_type, "blob_path": blob_path}
        # Let fakeredis/redis manage TTL
        self.r.set(self._idx_key(key), json.dumps(meta), ex=int(ttl_seconds))
        return {"key": key, "ttl": int(ttl_seconds), **meta}

    
    def get_TTL(self, logical_key: str) -> Optional[int]:
        key = _sha256(logical_key)
        ttl = self.r.ttl(self._idx_key(key))
        # redis/fakeredis: -2 = no key, -1 = no expire
        if ttl is None or ttl < 0:
            return None
        return int(ttl)


    def get_if_fresh(self, logical_key: str) -> Optional[Dict[str, Any]]:
        key = _sha256(logical_key)
        raw = self.r.get(self._idx_key(key))
        if not raw:
            return None
        try:
            meta = json.loads(raw)
        except Exception:
            return None

        blob_path = meta.get("blob_path")
        if not blob_path or not os.path.exists(blob_path):
            return None

        payload = self._read_blob(blob_path)
        return {
            "key": key,
            "payload": payload,
            "content_type": meta.get("content_type", "text/plain"),
        }


    def delete(self, logical_key: str) -> bool:
        key = _sha256(logical_key)
        raw = self.r.get(self._idx_key(key))
        if raw:
            try:
                meta = json.loads(raw)
                blob_path = meta.get("blob_path")
                if blob_path and os.path.exists(blob_path):
                    os.remove(blob_path)
            except Exception:
                pass
        self.r.delete(self._idx_key(key))
        return True

    # --- blob I/O ---
    def _write_blob(self, path: str, payload: str | bytes) -> None:
        mode = "wb" if isinstance(payload, (bytes, bytearray)) else "w"
        with open(path, mode) as f:
            f.write(payload)

    def _read_blob(self, path: str) -> str | bytes:
        # Attempt to detect binary vs text; default to text.
        try:
            with open(path, "rb") as f:
                data = f.read()
            # Heuristic: if data contains non-text bytes, return as bytes
            if b"\x00" in data:
                return data
            return data.decode("utf-8")
        except UnicodeDecodeError:
            with open(path, "rb") as f:
                return f.read()

