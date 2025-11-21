import io
import gzip
from typing import Tuple


def compress_bytes(data: bytes, level: int = 6) -> bytes:
    out = io.BytesIO()
    with gzip.GzipFile(fileobj=out, mode="wb", compresslevel=level) as f:
        f.write(data)
    return out.getvalue()


def decompress_bytes(data: bytes) -> bytes:
    with gzip.GzipFile(fileobj=io.BytesIO(data), mode="rb") as f:
        return f.read()

