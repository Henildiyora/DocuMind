"""Background incremental indexing while you edit (optional dependency)."""

from __future__ import annotations

import threading
import time
from pathlib import Path

from rich.console import Console

from .config import Config


def run_watch(
    project_root: Path,
    cfg: Config,
    *,
    debounce_sec: float = 0.5,
    console: Console | None = None,
) -> None:
    """Watch source files and run incremental ``build_or_update`` after quiet period."""
    try:
        from watchdog.events import FileSystemEventHandler
        from watchdog.observers import Observer
    except ImportError as exc:  # pragma: no cover - exercised via install path
        raise RuntimeError(
            "Install watchdog to use `documind watch`:\n"
            "  pip install 'documind[watch]'\n"
            "  # or: pip install 'watchdog>=4'"
        ) from exc

    from .index import DocuMindIndex

    console = console or Console()
    idx = DocuMindIndex(project_root, cfg)
    lock = threading.Lock()
    pending: list[str] = []
    timer: threading.Timer | None = None

    def flush() -> None:
        nonlocal timer
        with lock:
            if not pending:
                return
            pending.clear()
            timer = None
        console.print("[dim]Indexing…[/dim]")
        try:
            idx.build_or_update()
        except TimeoutError:
            console.print(
                "[yellow]Another indexing process holds the lock.[/yellow] "
                "Close the other `documind index` / `documind watch` or wait."
            )
            return
        except Exception as exc:  # noqa: BLE001 — surface to user
            console.print(f"[red]Index failed:[/red] {exc}")
            return
        console.print("[green]Index updated.[/green]")

    def schedule(_reason: str) -> None:
        nonlocal timer
        with lock:
            if timer is not None:
                timer.cancel()
            timer = threading.Timer(debounce_sec, flush)
            timer.daemon = True
            timer.start()

    class Handler(FileSystemEventHandler):
        def _handle(self, event) -> None:  # noqa: ANN001
            if event.is_directory:
                return
            path = getattr(event, "src_path", None) or getattr(event, "dest_path", None)
            if path is None:
                return
            p = Path(str(path))
            if p.name.startswith("."):
                return
            if ".documind" in p.parts:
                return
            with lock:
                pending.append(str(p))
            schedule("change")

        def on_modified(self, event) -> None:
            self._handle(event)

        def on_created(self, event) -> None:
            self._handle(event)

        def on_moved(self, event) -> None:
            self._handle(event)

        def on_deleted(self, event) -> None:
            self._handle(event)

    observer = Observer()
    observer.schedule(Handler(), str(project_root.resolve()), recursive=True)
    observer.start()
    console.print(
        f"[bold]Watching[/bold] {project_root.resolve()} "
        f"[dim](debounce {debounce_sec}s; Ctrl+C to stop)[/dim]"
    )
    try:
        while observer.is_alive():
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        observer.stop()
        observer.join(timeout=5)
        idx.close()
