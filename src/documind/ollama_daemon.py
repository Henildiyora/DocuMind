"""Helpers to make sure the Ollama daemon is running.

We try hard to avoid forcing the user to "open another terminal and run
`ollama serve`". Order of preference:

1. Daemon already reachable -> done.
2. `brew services start ollama`  (macOS, when brew is present).
3. `systemctl --user start ollama` (Linux, when systemd is available).
4. Spawn `ollama serve` as a detached background process.

Each attempt is followed by short polling so we return True as soon as the
daemon responds.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import time
from dataclasses import dataclass

from .config import Config
from .llm import OllamaClient


@dataclass
class DaemonStatus:
    """Result of a `ensure_daemon_running` attempt."""

    running: bool
    how: str  # "already" | "brew" | "systemctl" | "spawn" | "failed" | "missing"


def _ping(cfg: Config) -> bool:
    return OllamaClient(cfg).ping()


def _poll_until_up(cfg: Config, *, timeout_s: float = 6.0, interval_s: float = 0.25) -> bool:
    """Poll `ping()` until it returns True or the timeout elapses."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if _ping(cfg):
            return True
        time.sleep(interval_s)
    return _ping(cfg)


def _run_silent(cmd: list[str], timeout_s: float = 10.0) -> bool:
    """Run a command silently; return True on exit code 0."""
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout_s,
            check=False,
        )
        return proc.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def _try_brew_service() -> bool:
    if not shutil.which("brew"):
        return False
    return _run_silent(["brew", "services", "start", "ollama"], timeout_s=15.0)


def _try_systemctl_user() -> bool:
    if not shutil.which("systemctl"):
        return False
    return _run_silent(["systemctl", "--user", "start", "ollama"], timeout_s=10.0)


def _spawn_detached_serve() -> bool:
    """Spawn `ollama serve` so the daemon outlives this process."""
    if not shutil.which("ollama"):
        return False
    try:
        subprocess.Popen(  # noqa: S603 - explicit detached spawn
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
        return True
    except OSError:
        return False


def ensure_daemon_running(cfg: Config) -> DaemonStatus:
    """Make the Ollama daemon reachable, starting it if necessary.

    Non-interactive: no prompts, no stdin reads. Callers that want a
    confirmation prompt should do that themselves and then call this.
    """
    if _ping(cfg):
        return DaemonStatus(True, "already")

    if not shutil.which("ollama"):
        return DaemonStatus(False, "missing")

    system = platform.system()

    if system == "Darwin" and _try_brew_service() and _poll_until_up(cfg):
        return DaemonStatus(True, "brew")

    if system == "Linux" and _try_systemctl_user() and _poll_until_up(cfg):
        return DaemonStatus(True, "systemctl")

    if _spawn_detached_serve() and _poll_until_up(cfg):
        return DaemonStatus(True, "spawn")

    return DaemonStatus(False, "failed")


def install_hint() -> str:
    """Return a one-line install command for the current OS."""
    system = platform.system()
    if system == "Darwin":
        return "brew install ollama   # or https://ollama.com/download"
    if system == "Linux":
        return "curl -fsSL https://ollama.com/install.sh | sh"
    return "See https://ollama.com/download"
