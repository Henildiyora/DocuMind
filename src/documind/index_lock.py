"""Exclusive project lock so only one `documind index` runs at a time.

The lock file lives next to `.documind/` (not inside it) so `destroy()` /
`reset` never deletes an active lock inode while another process waits.
"""

from __future__ import annotations

import errno
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from pathlib import Path


def _lock_path(project_root: Path) -> Path:
    return project_root.resolve() / ".documind-index.lock"


@contextmanager
def index_write_lock(project_root: Path, *, timeout_sec: float = 600.0) -> Iterator[None]:
    """Acquire an exclusive lock for indexing ``project_root``."""
    path = _lock_path(project_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    f = open(path, "a+b")  # noqa: SIM115 — closed in finally
    start = time.monotonic()
    try:
        if sys.platform == "win32":
            import msvcrt

            while True:
                try:
                    f.seek(0)
                    msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError as exc:
                    if exc.errno not in (errno.EACCES, errno.EAGAIN, 13):
                        raise
                    if time.monotonic() - start > timeout_sec:
                        msg = f"Could not acquire index lock at {path} (timeout {timeout_sec}s)"
                        raise TimeoutError(msg) from exc
                    time.sleep(0.05)
        else:
            import fcntl

            while True:
                try:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError as exc:
                    if time.monotonic() - start > timeout_sec:
                        msg = f"Could not acquire index lock at {path} (timeout {timeout_sec}s)"
                        raise TimeoutError(msg) from exc
                    time.sleep(0.05)
        yield
    finally:
        if sys.platform == "win32":
            import msvcrt

            with suppress(OSError):
                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            with suppress(OSError):
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        f.close()
