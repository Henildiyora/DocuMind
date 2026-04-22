"""DocuMind Typer-based CLI."""

from __future__ import annotations

import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.prompt import Confirm
from rich.syntax import Syntax
from rich.table import Table

from . import __version__
from .config import Config, load_config, write_default_config
from .index import DocuMindIndex
from .llm import LLMError, OllamaClient
from .ollama_daemon import ensure_daemon_running, install_hint
from .prompts import build_messages
from .search import format_snippet, hits_to_context, search

app = typer.Typer(
    name="documind",
    help="Fast, typo-tolerant semantic + keyword search for your codebase.",
    add_completion=False,
    no_args_is_help=True,
    rich_markup_mode="rich",
)
console = Console()


# --------------------------------------------------------------------- shared


def _resolve_root(path: Path | None) -> Path:
    return (path or Path.cwd()).resolve()


def _build_index_with_progress(idx: DocuMindIndex):
    """Run a full incremental index build with a Rich progress UI.

    Returns the resulting :class:`IndexStats`.
    """
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    ) as progress:
        tasks: dict[str, int] = {}

        def on_progress(phase: str, done: int, total: int) -> None:
            label = {
                "chunking": "Chunking",
                "embedding": "Embedding",
                "bm25": "BM25",
            }.get(phase, phase)
            if phase not in tasks:
                tasks[phase] = progress.add_task(label, total=max(total, 1))
            progress.update(tasks[phase], completed=done, total=max(total, 1))

        return idx.build_or_update(progress=on_progress)


def _ensure_index(
    idx: DocuMindIndex,
    *,
    auto_index: bool | None,
) -> bool:
    """If the index is missing, optionally build it. Returns True if ready.

    `auto_index=True`  -> always build without asking.
    `auto_index=False` -> never build; just return False.
    `auto_index=None`  -> prompt if attached to a TTY; otherwise build.
    """
    if idx.exists():
        return True

    if auto_index is False:
        console.print(
            "[red]No index found.[/red] Run [bold]documind index[/bold] first."
        )
        return False

    should_build = auto_index is True
    if auto_index is None:
        if sys.stdin.isatty():
            console.print(
                f"[yellow]No index found[/yellow] at {idx.index_dir}."
            )
            should_build = Confirm.ask("Index this project now?", default=True)
        else:
            should_build = True

    if not should_build:
        console.print("Run [bold]documind index[/bold] when you're ready.")
        return False

    console.print(f"[bold]Indexing[/bold] {idx.project_root}")
    _build_index_with_progress(idx)
    return True


def _llm_ready_nonblocking(cfg: Config) -> OllamaClient | None:
    """Return a ready ``OllamaClient`` iff the daemon is up and the model is pulled.

    Used only by ``documind search`` so the zero-config path stays
    zero-overhead. This function never prompts, never auto-starts the
    daemon, and never prints: it silently returns None if the LLM isn't
    ready right now.
    """
    try:
        llm = OllamaClient(cfg)
        if not llm.ping():
            return None
        if not llm.model_available():
            return None
        return llm
    except Exception:
        return None


def _ensure_llm_ready(cfg: Config) -> OllamaClient | None:
    """Make Ollama reachable and the configured model available.

    Returns a ready `OllamaClient` or None on failure (after printing a
    clear user-facing message). Never asks the user to open a second
    terminal -- we try to start Ollama ourselves first.
    """
    import shutil

    if shutil.which("ollama") is None:
        console.print(
            f"[yellow]Ollama isn't installed.[/yellow] Install it with:\n"
            f"  [bold]{install_hint()}[/bold]\n"
            "Then retry. (Tip: `documind search` works without any model.)"
        )
        return None

    status = ensure_daemon_running(cfg)
    if not status.running:
        console.print(
            "[red]Couldn't start Ollama automatically.[/red] "
            "Try: [bold]ollama serve[/bold] in another terminal, then retry."
        )
        return None
    if status.how != "already":
        console.print(f"[dim]Started Ollama via {status.how}.[/dim]")

    llm = OllamaClient(cfg)
    if not llm.model_available():
        console.print(
            f"[yellow]Model {cfg.model!r} isn't pulled yet.[/yellow] "
            f"Run: [bold]documind setup --pull[/bold] "
            f"or [bold]ollama pull {cfg.model}[/bold]"
        )
        return None
    return llm


def _make_cfg(model: str | None, k: int | None) -> Config:
    overrides: dict = {}
    if model:
        overrides["model"] = model
    if k:
        overrides["top_k"] = k
    return load_config(overrides or None)


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"documind {__version__}")
        raise typer.Exit()


@app.callback()
def _root(
    version: bool = typer.Option(
        False, "--version", "-V", callback=_version_callback, is_eager=True,
        help="Show version and exit.",
    ),
) -> None:
    """DocuMind: pure-local hybrid search for any project."""


# --------------------------------------------------------------------- index


@app.command("index")
def cmd_index(
    path: Path | None = typer.Argument(None, help="Project root (default: cwd)."),
    rebuild: bool = typer.Option(False, "--rebuild", help="Delete and rebuild from scratch."),
) -> None:
    """Index a project (incremental by default)."""
    root = _resolve_root(path)
    cfg = load_config()
    idx = DocuMindIndex(root, cfg)

    if rebuild:
        console.print(f"[yellow]Rebuilding index at[/yellow] {idx.index_dir}")
        idx.destroy()

    console.print(f"[bold]Indexing[/bold] {root}")
    stats = _build_index_with_progress(idx)
    idx.close()

    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_row("Scanned files",  str(stats.scanned_files))
    table.add_row("New",            f"[green]{stats.new_files}[/green]")
    table.add_row("Changed",        f"[yellow]{stats.changed_files}[/yellow]")
    table.add_row("Unchanged",      str(stats.unchanged_files))
    table.add_row("Removed",        f"[red]{stats.removed_files}[/red]")
    table.add_row("Embedded chunks", str(stats.embedded_chunks))
    table.add_row("Total chunks",   str(stats.total_chunks))
    console.print(table)
    console.print(f"[green]Index ready[/green] at {idx.index_dir}")


# --------------------------------------------------------------------- search


def _print_hits(hits, cfg: Config, *, show_code: bool, compact: bool) -> None:
    """Render ranked search hits.

    Parameters
    ----------
    hits:
        Retrieved ``SearchHit`` records to render.
    cfg:
        Active ``Config`` (passed to ``format_snippet`` for snippet trimming).
    show_code:
        If True, print the full code body under each header.
    compact:
        If True, use the tight "sources" layout (one indented line per hit).
        If False, use the full-width header with RRF debug info.
    """
    for rank, hit in enumerate(hits, start=1):
        bm25 = f"bm25#{hit.bm25_rank}" if hit.bm25_rank else "--"
        vec = f"vec#{hit.vector_rank}" if hit.vector_rank else "--"
        if compact:
            header = (
                f"  [bold cyan]{rank:>2}[/bold cyan]  "
                f"[green]{hit.rel_path}[/green]"
                f":[magenta]{hit.start_line}-{hit.end_line}[/magenta]  "
                f"[dim]({bm25}, {vec})[/dim]"
            )
        else:
            header = (
                f"[bold cyan]{rank:>2}[/bold cyan] "
                f"[green]{hit.rel_path}[/green]"
                f":[magenta]{hit.start_line}-{hit.end_line}[/magenta] "
                f"[dim]({bm25}, {vec}, rrf={hit.score:.4f})[/dim]"
            )
        console.print(header)
        if show_code:
            snippet = format_snippet(hit, cfg)
            lang = hit.language if hit.language not in {"text", "pdf"} else "text"
            try:
                console.print(Syntax(snippet, lang, line_numbers=False, theme="ansi_dark", word_wrap=True))
            except Exception:
                console.print(snippet)
            console.print()


@app.command("search")
def cmd_search(
    query: str = typer.Argument(..., help="Search query or natural-language question."),
    path: Path | None = typer.Option(None, "--path", "-p", help="Project root."),
    k: int | None = typer.Option(None, "--k", "-k", help="Number of results."),
    summary: bool | None = typer.Option(
        None,
        "--summary/--no-summary",
        help="Synthesize a natural-language answer if a local model is available (default: auto).",
    ),
    show_code: bool = typer.Option(
        False,
        "--code/--no-code",
        help="Print full code snippets (default: compact file refs when summarizing).",
    ),
    auto_index: bool | None = typer.Option(
        None,
        "--auto-index/--no-auto-index",
        help="Build the index on the fly if missing (default: prompt when interactive).",
    ),
) -> None:
    """Fast hybrid search. Answers in natural language when a local model is available.

    Zero-config path: with no model installed, this is a pure BM25 + vector
    search over your project -- 100% free, 100% local, no API keys.
    """
    root = _resolve_root(path)
    cfg = _make_cfg(None, k)
    idx = DocuMindIndex(root, cfg)
    if not _ensure_index(idx, auto_index=auto_index):
        raise typer.Exit(1)

    hits = search(idx, query, cfg)
    if not hits:
        console.print("[yellow]No matches.[/yellow]")
        idx.close()
        return

    llm = None if summary is False else _llm_ready_nonblocking(cfg)

    if llm is not None:
        console.print(
            f"[bold]Answer[/bold]  [dim](local model: {cfg.model}, 100% free)[/dim]"
        )
        messages = build_messages(query, hits_to_context(hits))
        buffer = ""
        try:
            with Live(Markdown(""), console=console, refresh_per_second=20) as live:
                for tok in llm.chat_stream(messages):
                    buffer += tok
                    live.update(Markdown(buffer))
        except LLMError as exc:
            console.print(f"[dim]summary failed: {exc}[/dim]")
        console.print()
        console.print("[bold]Sources[/bold]")
        _print_hits(hits, cfg, show_code=show_code, compact=True)
    else:
        if summary is True:
            console.print(
                "[yellow]No local model available for a summary.[/yellow] "
                "Run [bold]documind setup[/bold] (free, local) or drop [bold]--summary[/bold]."
            )
        console.print(f"[bold]Snippets from[/bold] [cyan]{root}[/cyan]")
        _print_hits(hits, cfg, show_code=True, compact=False)
        if summary is None:
            console.print(
                "[dim]Tip: run `documind setup` for a free, local natural-language answer on top.[/dim]"
            )

    idx.close()


# ---------------------------------------------------------------------- ask


@app.command("ask")
def cmd_ask(
    query: str = typer.Argument(..., help="Question to ask."),
    path: Path | None = typer.Option(None, "--path", "-p", help="Project root."),
    k: int | None = typer.Option(None, "--k", "-k", help="Number of snippets."),
    model: str | None = typer.Option(None, "--model", "-m", help="Ollama model override."),
    no_llm: bool = typer.Option(False, "--no-llm", help="Print ranked hits only (skip LLM)."),
    auto_index: bool | None = typer.Option(
        None,
        "--auto-index/--no-auto-index",
        help="Build the index on the fly if missing (default: prompt when interactive).",
    ),
) -> None:
    """Ask a grounded question. Retrieves snippets and synthesizes with Gemma."""
    root = _resolve_root(path)
    cfg = _make_cfg(model, k)
    idx = DocuMindIndex(root, cfg)
    if not _ensure_index(idx, auto_index=auto_index):
        raise typer.Exit(1)

    hits = search(idx, query, cfg)
    if not hits:
        console.print("[yellow]No matches.[/yellow]")
        raise typer.Exit(0)

    if no_llm:
        for rank, hit in enumerate(hits, start=1):
            console.print(f"[bold]{rank}.[/bold] {hit.rel_path}:{hit.start_line}-{hit.end_line}")
            console.print(format_snippet(hit, cfg))
            console.print()
        return

    llm = _ensure_llm_ready(cfg)
    if llm is None:
        raise typer.Exit(2)

    context = hits_to_context(hits)
    messages = build_messages(query, context)

    buffer = ""
    try:
        with Live(Markdown(""), console=console, refresh_per_second=20) as live:
            for tok in llm.chat_stream(messages):
                buffer += tok
                live.update(Markdown(buffer))
    except LLMError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(3) from exc

    refs = ", ".join(f"{h.rel_path}:{h.start_line}-{h.end_line}" for h in hits)
    console.print(f"\n[dim]sources: {refs}[/dim]")
    idx.close()


# --------------------------------------------------------------------- chat


@app.command("chat")
def cmd_chat(
    path: Path | None = typer.Option(None, "--path", "-p", help="Project root."),
    model: str | None = typer.Option(None, "--model", "-m", help="Ollama model override."),
    k: int | None = typer.Option(None, "--k", "-k", help="Snippets per question."),
) -> None:
    """Start an interactive chat grounded in your project."""
    from .chat import run_chat

    root = _resolve_root(path)
    cfg = _make_cfg(model, k)
    run_chat(root, cfg)


# ---------------------------------------------------------------------- setup


@app.command("setup")
def cmd_setup(
    path: Path | None = typer.Option(None, "--path", "-p", help="Project to scan for the recommendation."),
    tier: str | None = typer.Option(None, "--tier", help="Force a tier: tiny, small, or deep."),
    model: str | None = typer.Option(None, "--model", "-m", help="Force a specific Ollama model tag (e.g. qwen2.5-coder:14b)."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Accept the recommendation without prompting."),
    pull: bool | None = typer.Option(
        None,
        "--pull/--no-pull",
        help="Pull the model via Ollama. Default: ask (or skip in non-interactive shells).",
    ),
) -> None:
    """Pick a local model for `documind ask` / `documind chat`.

    Search and index never need a model, so this command is optional.
    It saves your model preference and can (optionally) pull it via Ollama.
    """
    from .setup import run_setup

    root = _resolve_root(path)
    code = run_setup(root, tier=tier, model=model, yes=yes, pull=pull)
    if code != 0:
        raise typer.Exit(code)


# --------------------------------------------------------------------- doctor


@app.command("doctor")
def cmd_doctor(
    path: Path | None = typer.Option(None, "--path", "-p", help="Project root."),
    pull: bool = typer.Option(False, "--pull", help="Pull the configured model if missing."),
    write_config: bool = typer.Option(
        False, "--write-config", help="Write ~/.config/documind/config.toml if missing."
    ),
) -> None:
    """Check your environment: Ollama, model, and index state."""
    root = _resolve_root(path)
    cfg = load_config()

    table = Table(title="DocuMind doctor", show_header=True, header_style="bold")
    table.add_column("Check")
    table.add_column("Status")
    table.add_column("Detail")

    # Config
    if write_config:
        p = write_default_config()
        table.add_row("config", "[green]written[/green]", str(p))
    else:
        table.add_row("config", "[green]ok[/green]", f"model={cfg.model}, emb={cfg.embedding_model}")

    # Ollama (best-effort auto-start so the table reflects reality)
    llm = OllamaClient(cfg)
    if llm.ping():
        table.add_row("ollama daemon", "[green]ok[/green]", cfg.ollama_base_url)
    else:
        status = ensure_daemon_running(cfg)
        if status.running:
            table.add_row(
                "ollama daemon",
                "[green]ok[/green]",
                f"{cfg.ollama_base_url} (started via {status.how})",
            )
        elif status.how == "missing":
            table.add_row(
                "ollama daemon",
                "[red]missing[/red]",
                f"Install: {install_hint()}",
            )
        else:
            table.add_row(
                "ollama daemon",
                "[red]down[/red]",
                f"Run: ollama serve ({cfg.ollama_base_url})",
            )

    # Model
    if llm.model_available():
        table.add_row("model", "[green]ok[/green]", cfg.model)
    else:
        if pull:
            try:
                console.print(f"Pulling {cfg.model}...")
                llm.pull()
                table.add_row("model", "[green]pulled[/green]", cfg.model)
            except LLMError as exc:
                table.add_row("model", "[red]error[/red]", str(exc))
        else:
            table.add_row(
                "model", "[yellow]missing[/yellow]", f"Run: ollama pull {cfg.model}"
            )

    # Index
    idx = DocuMindIndex(root, cfg)
    if idx.exists():
        chunks = len(idx.all_chunks())
        table.add_row("index", "[green]ok[/green]", f"{chunks} chunks at {idx.index_dir}")
    else:
        table.add_row("index", "[yellow]none[/yellow]", f"Run: documind index {root}")
    idx.close()

    console.print(table)


# --------------------------------------------------------------------- reset


@app.command("reset")
def cmd_reset(
    path: Path | None = typer.Option(None, "--path", "-p", help="Project root."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation."),
) -> None:
    """Delete the project's index directory (`.documind/`)."""
    root = _resolve_root(path)
    cfg = load_config()
    idx = DocuMindIndex(root, cfg)
    if not idx.index_dir.exists():
        console.print("[yellow]Nothing to delete.[/yellow]")
        return
    if not yes:
        confirm = typer.confirm(f"Delete {idx.index_dir}?", default=False)
        if not confirm:
            raise typer.Exit(1)
    idx.destroy()
    console.print(f"[green]Removed[/green] {idx.index_dir}")


def main() -> None:
    try:
        app()
    except KeyboardInterrupt:
        console.print()
        sys.exit(130)


if __name__ == "__main__":
    main()
