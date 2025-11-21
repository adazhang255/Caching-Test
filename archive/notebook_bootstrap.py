import os
import subprocess
import time
from typing import Optional

_PROC: Optional[subprocess.Popen] = None


def start_controller(
    config_path: str = "src/config.yaml",
    host: str = "127.0.0.1",
    port: int = 9000,
    monitor_port: int = 9001,
    log_path: str = "controller.log",
    env_offload_disable: Optional[str] = None,
) -> int:
    """Start lmcache_controller in the background for notebooks.

    Returns the controller PID. Writes logs to log_path.
    Set env_offload_disable="1" to force GPU-only (no moves), or leave None to keep default behavior.
    """
    global _PROC
    if _PROC and _PROC.poll() is None:
        return _PROC.pid

    env = os.environ.copy()
    env["LMCACHE_CONFIG_FILE"] = config_path
    if env_offload_disable is not None:
        env["LMCACHE_DISABLE_OFFLOAD"] = env_offload_disable

    cmd = [
        "lmcache_controller",
        "--host", str(host),
        "--port", str(port),
        "--monitor-port", str(monitor_port),
        "--config", config_path,
    ]

    logf = open(log_path, "ab", buffering=0)
    _PROC = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT)
    # small grace period
    time.sleep(1.0)
    return _PROC.pid


def stop_controller(timeout: float = 3.0) -> None:
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


def controller_running() -> bool:
    """Return True if controller process is alive."""
    return _PROC is not None and _PROC.poll() is None

