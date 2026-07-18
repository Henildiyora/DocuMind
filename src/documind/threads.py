"""Persistent named chat threads under ``.documind/chats/``."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .config import Config, index_dir_for, load_config

_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]+$")
DEFAULT_THREAD = "default"


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sanitize_thread_name(name: str) -> str:
    """Validate and return a safe thread name."""
    cleaned = (name or "").strip()
    if not cleaned or not _NAME_RE.match(cleaned):
        raise ValueError(
            f"Invalid thread name {name!r}. Use letters, digits, _ or - only."
        )
    return cleaned


def chats_dir(project_root: Path, cfg: Config | None = None) -> Path:
    """Return ``<project>/.documind/chats``."""
    cfg = cfg or load_config()
    return index_dir_for(project_root, cfg) / "chats"


@dataclass
class Thread:
    """A named conversation stored as JSON."""

    name: str
    created_at: str = field(default_factory=_utc_now)
    updated_at: str = field(default_factory=_utc_now)
    messages: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "messages": self.messages,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Thread:
        return cls(
            name=str(data.get("name", DEFAULT_THREAD)),
            created_at=str(data.get("created_at") or _utc_now()),
            updated_at=str(data.get("updated_at") or _utc_now()),
            messages=list(data.get("messages") or []),
        )

    def preview(self, limit: int = 60) -> str:
        """Last message content preview for ``/threads``."""
        if not self.messages:
            return "(empty)"
        text = str(self.messages[-1].get("content") or "").strip().replace("\n", " ")
        if len(text) > limit:
            return text[: limit - 1] + "…"
        return text or "(empty)"

    def history_window(self, turns: int) -> list[dict]:
        """Return the last ``turns`` user+assistant pairs as Ollama messages."""
        n = max(0, int(turns)) * 2
        if n <= 0:
            return []
        window = self.messages[-n:]
        return [{"role": m["role"], "content": m["content"]} for m in window]


def _thread_path(project_root: Path, name: str, cfg: Config | None = None) -> Path:
    safe = sanitize_thread_name(name)
    return chats_dir(project_root, cfg) / f"{safe}.json"


def load_thread(
    project_root: Path,
    name: str = DEFAULT_THREAD,
    cfg: Config | None = None,
) -> Thread:
    """Load a thread from disk, or create an empty in-memory one."""
    path = _thread_path(project_root, name, cfg)
    if not path.exists():
        return Thread(name=sanitize_thread_name(name))
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        thread = Thread.from_dict(data)
        thread.name = sanitize_thread_name(name)
        return thread
    except (OSError, json.JSONDecodeError, ValueError):
        return Thread(name=sanitize_thread_name(name))


def save_thread(
    project_root: Path,
    thread: Thread,
    cfg: Config | None = None,
) -> Path:
    """Persist ``thread`` to ``.documind/chats/<name>.json``."""
    path = _thread_path(project_root, thread.name, cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    thread.updated_at = _utc_now()
    path.write_text(json.dumps(thread.to_dict(), indent=2) + "\n", encoding="utf-8")
    return path


def append_turn(
    project_root: Path,
    name: str,
    user: str,
    assistant: str,
    cfg: Config | None = None,
) -> Thread:
    """Append a user/assistant turn and save."""
    thread = load_thread(project_root, name, cfg)
    now = _utc_now()
    thread.messages.append({"role": "user", "content": user, "ts": now})
    thread.messages.append({"role": "assistant", "content": assistant, "ts": now})
    save_thread(project_root, thread, cfg)
    return thread


def list_threads(project_root: Path, cfg: Config | None = None) -> list[Thread]:
    """Return all threads sorted by ``updated_at`` descending."""
    root = chats_dir(project_root, cfg)
    if not root.exists():
        return []
    threads: list[Thread] = []
    for path in sorted(root.glob("*.json")):
        try:
            threads.append(load_thread(project_root, path.stem, cfg))
        except ValueError:
            continue
    threads.sort(key=lambda t: t.updated_at, reverse=True)
    return threads


def rename_thread(
    project_root: Path,
    old: str,
    new: str,
    cfg: Config | None = None,
) -> Thread:
    """Rename a thread file and update its name field."""
    old_safe = sanitize_thread_name(old)
    new_safe = sanitize_thread_name(new)
    if old_safe == new_safe:
        return load_thread(project_root, old_safe, cfg)
    src = _thread_path(project_root, old_safe, cfg)
    dst = _thread_path(project_root, new_safe, cfg)
    if not src.exists():
        raise FileNotFoundError(f"Thread {old_safe!r} does not exist.")
    if dst.exists():
        raise FileExistsError(f"Thread {new_safe!r} already exists.")
    thread = load_thread(project_root, old_safe, cfg)
    thread.name = new_safe
    save_thread(project_root, thread, cfg)
    src.unlink(missing_ok=True)
    return thread


def delete_thread(
    project_root: Path,
    name: str,
    cfg: Config | None = None,
) -> None:
    """Delete a thread file if present."""
    path = _thread_path(project_root, name, cfg)
    if path.exists():
        path.unlink()


def ensure_thread(
    project_root: Path,
    name: str,
    cfg: Config | None = None,
) -> Thread:
    """Load or create and persist an empty thread."""
    thread = load_thread(project_root, name, cfg)
    path = _thread_path(project_root, thread.name, cfg)
    if not path.exists():
        save_thread(project_root, thread, cfg)
    return thread
