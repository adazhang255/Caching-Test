from typing import Any, Dict, List, Optional
import os
import subprocess
import time

import requests

_PROC: Optional[subprocess.Popen] = None


class LMCacheController:
    """HTTP client + lightweight process manager for the LMCache API server."""

    def __init__(self, host: str = "127.0.0.1", port: int = 9000, model: str = "gemma-3-270m"):
        self.host = host
        self.port = port
        self.url = f"http://{host}:{port}"
        self.model = model
        self.vllm_url = "http://{host}:8000"

    def start_controller(
        self,
        config_path: str = "src/config.yaml",
        host: str = "127.0.0.1",
        port: int = 9000,
        log_path: str = "controller.log",
        env_offload_disable: Optional[str] = None,
        python_executable: str = "python3",
    ) -> int:
        """Start the LMCache API server via `python -m lmcache.v1.api_server`.

        Returns the controller PID. Writes logs to ``log_path``.
        Set ``env_offload_disable=\"1\"`` to force GPU-only (no offload),
        or leave ``None`` to keep default behavior.
        """
        global _PROC
        if _PROC and _PROC.poll() is None:
            return _PROC.pid

        env = os.environ.copy()
        env["LMCACHE_CONFIG_FILE"] = config_path
        if env_offload_disable is not None:
            env["LMCACHE_DISABLE_OFFLOAD"] = env_offload_disable

        cmd = [
            python_executable,
            "-m",
            "lmcache.v1.api_server",
            "--host",
            str(host),
            "--port",
            str(port),
        ]

        logf = open(log_path, "ab", buffering=0)
        _PROC = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env)
        time.sleep(1.0)
        return _PROC.pid

    def stop_controller(self, timeout: float = 3.0) -> None:
        """Stop the controller if running."""
        global _PROC
        if not _PROC:
            return
        if _PROC.poll() is None:
            _PROC.terminate()
            try:
                _PROC.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                _PROC.kill()
        _PROC = None

    def controller_running(self) -> bool:
        """Return True if controller process is alive."""
        return _PROC is not None and _PROC.poll() is None

    def health(self) -> Dict[str, Any]:
        try:
            payload = {"instance_id": "lmcache_instance"}
            r = requests.post(f"{self.url}/health", json=payload, timeout=2)
            r.raise_for_status()
            return {"ok": True, "data": r.json()}
        except Exception as e:
            return {"ok": False, "error": str(e)}
        

    def tokenize(self, prompt: str) -> List[int]:
        r = requests.post(self.vllm_url, json={"model": self.model, "prompt": prompt})
        r.raise_for_status()
        return r.json()

    def lookup(self, tokens: List[int]) -> Dict[str, Any]:
        r = requests.post(f"{self.url}/lookup", json={"tokens": tokens})
        r.raise_for_status()
        return r.json()

    def move(self, old_position: List[str], new_position: List[str]) -> Dict[str, Any]:
        r = requests.post(f"{self.url}/move", json={"old_position": old_position, "new_position": new_position})
        r.raise_for_status()
        return r.json()

    def hydrate_set(self, key: str, blob_bytes: bytes, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Shim for inserting a serialized KV into LMCache GPU via inference app.

        The upstream controller may not expose a direct insert API; the inference
        app should provide /kv/hydrate to accept serialized KV blobs and perform
        the insertion.
        """
        r = requests.post(
            f"{self.url}/kv/hydrate",
            files={"blob": ("kv.bin", blob_bytes)},
            data={"key": key, "meta": (meta or {})},
        )
        r.raise_for_status()
        return r.json()
