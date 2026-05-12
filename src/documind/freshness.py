"""Detect whether the working tree may be newer than the on-disk index."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console

from .chunker import iter_source_files
from .config import Config
from .index import DocuMindIndex


def max_mtime_source_tree(root: Path, cfg: Config) -> float | None:
    """Maximum ``st_mtime`` over all files DocuMind would index (stat only)."""
    root = root.resolve()
    best: float | None = None
    for p in iter_source_files(root, cfg.max_file_bytes):
        try:
            mt = p.stat().st_mtime
        except OSError:
            continue
        best = mt if best is None or mt > best else best
    return best


def index_may_be_stale(idx: DocuMindIndex, root: Path, cfg: Config) -> bool:
    """True if any indexed source file's mtime is newer than ``state.json`` snapshot."""
    state = idx.read_state()
    raw = state.get("max_mtime_at_index")
    if raw is None:
        return False
    try:
        indexed_max = float(raw)
    except (TypeError, ValueError):
        return False
    cur = max_mtime_source_tree(root, cfg)
    if cur is None:
        return False
    return cur > indexed_max + 1e-6


def maybe_warn_stale_index(
    console: Console,
    idx: DocuMindIndex,
    root: Path,
    cfg: Config,
    *,
    enabled: bool,
) -> None:
    if not enabled or not idx.exists():
        return
    if not index_may_be_stale(idx, root, cfg):
        return
    console.print(
        "[yellow]Your working tree looks newer than the last index.[/yellow] "
        "Run [bold]documind index[/bold] or [bold]documind watch[/bold] so search matches disk. "
        "[dim](Suppress with --no-stale-warn.)[/dim]"
    )
