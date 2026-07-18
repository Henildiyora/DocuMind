"""DocuMind Typer-based CLI."""

from __future__ import annotations

import contextlib
import sys
import traceback
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
from .config import Config, load_config, update_user_config, write_default_config
from .errors import (
    DocuMindError,
    IndexMissing,
    ModelNotPulled,
    OllamaNotInstalled,
    OllamaNotRunning,
    format_user_error,
)
from .freshness import maybe_warn_stale_index
from .index import DocuMindIndex
from .llm import LLMError, OllamaClient
from .ollama_daemon import ensure_daemon_running, install_hint, ollama_installed
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

# Set by the root callback when --debug / -v is passed.
DEBUG: bool = False


def _stdin_is_tty() -> bool:
    """Return True when stdin is interactive (patchable in tests)."""
    return sys.stdin.isatty()


# --------------------------------------------------------------------- shared


def _resolve_root(path: Path | None) -> Path:
    return (path or Path.cwd()).resolve()


def _handle_user_error(exc: DocuMindError) -> None:
    """Print a friendly message; include traceback only when DEBUG is set."""
    console.print(format_user_error(exc))
    if DEBUG:
        console.print("[dim]--- debug traceback ---[/dim]")
        console.print(traceback.format_exc())
    raise typer.Exit(exc.exit_code) from None


def _build_index_with_progress(idx: DocuMindIndex, *, force_rehash: bool = False):
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

        return idx.build_or_update(progress=on_progress, force_rehash=force_rehash)


def _effective_auto_index(
    auto_index: bool | None,
    init_index: bool | None,
) -> bool | None:
    """``--init-index`` overrides ``--auto-index`` when either is set."""
    if init_index is not None:
        return init_index
    return auto_index


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
        raise IndexMissing()

    should_build = auto_index is True
    if auto_index is None:
        if _stdin_is_tty():
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
    try:
        _build_index_with_progress(idx)
    except TimeoutError:
        console.print(
            "[red]Another `documind index` (or `documind watch`) is running[/red] "
            f"and holds the lock at {idx.project_root / '.documind-index.lock'}."
        )
        return False
    return True


def _llm_ready_nonblocking(cfg: Config) -> OllamaClient | None:
    """Return a ready ``OllamaClient`` iff the daemon is up and the model is pulled.

    Used only by ``documind search`` so the zero-config path stays
    zero-overhead. This function never prompts, never auto-starts the
    daemon, and never prints: it silently returns None if the LLM isn't
    ready right now.
    """
    try:
        if not ollama_installed():
            return None
        llm = OllamaClient(cfg)
        if not llm.ping():
            return None
        if not llm.model_available():
            return None
        return llm
    except Exception:
        return None


def _ensure_llm_ready(cfg: Config) -> OllamaClient:
    """Make Ollama reachable and the configured model available.

    Raises a typed :class:`DocuMindError` on failure. Never asks the user
    to open a second terminal -- we try to start Ollama ourselves first.
    """
    if not ollama_installed():
        raise OllamaNotInstalled()

    status = ensure_daemon_running(cfg)
    if not status.running:
        raise OllamaNotRunning()
    if status.how != "already":
        console.print(f"[dim]Started Ollama via {status.how}.[/dim]")

    llm = OllamaClient(cfg)
    if not llm.model_available():
        raise ModelNotPulled(cfg.model)
    return llm


def _make_cfg(
    model: str | None,
    k: int | None,
    keep_alive: str | None = None,
) -> Config:
    overrides: dict = {}
    if model:
        overrides["model"] = model
    if k:
        overrides["top_k"] = k
    if keep_alive is not None:
        overrides["keep_alive"] = keep_alive
    return load_config(overrides or None)


def _maybe_offer_setup(root: Path, cfg: Config) -> None:
    """After a successful index, optionally run the setup flow inline."""
    if not _stdin_is_tty():
        return
    if cfg.setup_done or not cfg.offer_setup_after_index:
        return

    want = Confirm.ask(
        "Index ready. Want AI-powered Q&A on top of this?",
        default=True,
    )
    if not want:
        update_user_config({"offer_setup_after_index": False})
        console.print(
            "[dim]run `documind setup` anytime to enable ask/chat[/dim]"
        )
        return

    from .setup import run_setup

    run_setup(root)


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"documind {__version__}")
        raise typer.Exit()


@app.callback()
def _root(
    ctx: typer.Context,
    version: bool = typer.Option(
        False, "--version", "-V", callback=_version_callback, is_eager=True,
        help="Show version and exit.",
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        "-v",
        help="Show full stack traces for errors (default: friendly messages only).",
    ),
) -> None:
    """DocuMind: pure-local hybrid search for any project.

    Use ``-v`` / ``--debug`` when reporting a bug so the full traceback is shown.
    """
    global DEBUG
    DEBUG = bool(debug)
    ctx.ensure_object(dict)
    ctx.obj["debug"] = DEBUG


# --------------------------------------------------------------------- index


@app.command("index")
def cmd_index(
    path: Path | None = typer.Argument(None, help="Project root (default: cwd)."),
    rebuild: bool = typer.Option(False, "--rebuild", help="Delete and rebuild from scratch."),
    force_rehash: bool = typer.Option(
        False,
        "--force-rehash",
        help="Re-read and re-hash every file (ignore mtime/size fast path).",
    ),
) -> None:
    """Index a project (incremental by default).

    After the first successful index, if you have not run setup yet, DocuMind
    offers to enable AI-powered ask/chat inline (TTY only).
    """
    root = _resolve_root(path)
    cfg = load_config()
    idx = DocuMindIndex(root, cfg)

    if rebuild:
        console.print(f"[yellow]Rebuilding index at[/yellow] {idx.index_dir}")
        idx.destroy()

    console.print(f"[bold]Indexing[/bold] {root}")
    try:
        stats = _build_index_with_progress(idx, force_rehash=force_rehash)
    except TimeoutError:
        console.print(
            "[red]Another indexing process holds the project lock.[/red]\n"
            "Close the other terminal running `documind index` or `documind watch`, "
            f"or remove a stale lock file: {root / '.documind-index.lock'}"
        )
        raise typer.Exit(1) from None
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
    idle = (
        stats.new_files == 0
        and stats.changed_files == 0
        and stats.removed_files == 0
        and stats.embedded_chunks == 0
    )
    if idle and stats.total_chunks > 0:
        console.print(
            f"[green]Index already up to date[/green] "
            f"({stats.total_chunks} chunks) at {idx.index_dir}"
        )
    else:
        console.print(f"[green]Index ready[/green] at {idx.index_dir}")

    # Re-load config in case another process wrote it; offer setup if needed.
    _maybe_offer_setup(root, load_config())


@app.command("watch")
def cmd_watch(
    path: Path | None = typer.Option(None, "--path", "-p", help="Project root."),
    debounce: float = typer.Option(
        0.5,
        "--debounce",
        help="Seconds to wait after the last file change before indexing.",
    ),
) -> None:
    """Watch files under the project and run incremental indexing after edits.

    Requires: pip install 'documind[watch]' (brings in watchdog).
    """
    from .watch import run_watch

    root = _resolve_root(path)
    cfg = load_config()
    try:
        run_watch(root, cfg, debounce_sec=debounce, console=console)
    except RuntimeError as exc:
        console.print(f"[yellow]{exc}[/yellow]")
        raise typer.Exit(1) from exc


# --------------------------------------------------------------------- search


def _print_hits(hits, cfg: Config, *, show_code: bool, compact: bool) -> None:
    """Render ranked search hits."""
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
                console.print(
                    Syntax(snippet, lang, line_numbers=False, theme="ansi_dark", word_wrap=True)
                )
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
        help="Create index if missing only (does not refresh a stale index).",
    ),
    init_index: bool | None = typer.Option(
        None,
        "--init-index/--no-init-index",
        help="Alias for --auto-index / --no-auto-index.",
    ),
    stale_warn: bool = typer.Option(
        True,
        "--stale-warn/--no-stale-warn",
        help="Warn when the working tree looks newer than the last index.",
    ),
) -> None:
    """Fast hybrid search. Answers in natural language when a local model is available.

    Zero-config path: with no model installed, this is a pure BM25 + vector
    search over your project -- 100% free, 100% local, no API keys.
    """
    try:
        root = _resolve_root(path)
        cfg = _make_cfg(None, k)
        idx = DocuMindIndex(root, cfg)
        eff = _effective_auto_index(auto_index, init_index)
        if not _ensure_index(idx, auto_index=eff):
            raise typer.Exit(1)

        maybe_warn_stale_index(console, idx, root, cfg, enabled=stale_warn)

        hits = search(idx, query, cfg)
        if not hits:
            console.print("[yellow]No matches.[/yellow]")
            idx.close()
            return

        # Demote generated reports / boost README+entrypoints for overview Qs.
        # Keeps snippet-only search readable; summary path especially needs this.
        from .query_understand import refine_hits

        hits = refine_hits(hits, query, cfg)

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
    except DocuMindError as exc:
        _handle_user_error(exc)


# ---------------------------------------------------------------------- ask


@app.command("ask")
def cmd_ask(
    query: list[str] | None = typer.Argument(
        None,
        help=(
            "Question to ask (unquoted trailing words are joined). "
            "Omit the question to open interactive chat."
        ),
    ),
    path: Path | None = typer.Option(None, "--path", "-p", help="Project root."),
    k: int | None = typer.Option(None, "--k", "-k", help="Number of snippets."),
    model: str | None = typer.Option(None, "--model", "-m", help="Ollama model override."),
    keep_alive: str | None = typer.Option(
        None,
        "--keep-alive",
        help=(
            "How long Ollama keeps the model in RAM after answering "
            "(e.g. 5m, 0 to unload immediately). Lower idle RAM/CPU/energy use."
        ),
    ),
    no_llm: bool = typer.Option(False, "--no-llm", help="Print ranked hits only (skip LLM)."),
    no_clarify: bool = typer.Option(
        False,
        "--no-clarify",
        help="Skip interactive clarification when the query is ambiguous.",
    ),
    auto_index: bool | None = typer.Option(
        None,
        "--auto-index/--no-auto-index",
        help="Create index if missing only (does not refresh a stale index).",
    ),
    init_index: bool | None = typer.Option(
        None,
        "--init-index/--no-init-index",
        help="Alias for --auto-index / --no-auto-index.",
    ),
    stale_warn: bool = typer.Option(
        True,
        "--stale-warn/--no-stale-warn",
        help="Warn when the working tree looks newer than the last index.",
    ),
) -> None:
    """Ask a grounded question, or open chat when no question is given.

    Trailing words are joined, so quotes are optional::

        documind ask why does the rate limiter reset early

    With no question, this starts the same interactive REPL as ``documind chat``::

        documind ask
    """
    from .chat import run_chat
    from .query_understand import refine_hits, retrieve_for_question
    from .threads import append_turn

    try:
        joined = " ".join(query or []).strip()
        root = _resolve_root(path)
        cfg = _make_cfg(model, k, keep_alive)

        # Bare `documind ask` → conversation mode (same as `documind chat`).
        if not joined:
            run_chat(root, cfg)
            return

        idx = DocuMindIndex(root, cfg)
        eff = _effective_auto_index(auto_index, init_index)
        if not _ensure_index(idx, auto_index=eff):
            raise typer.Exit(1)

        maybe_warn_stale_index(console, idx, root, cfg, enabled=stale_warn)

        if no_llm:
            hits = refine_hits(search(idx, joined, cfg), joined, cfg)
            if not hits:
                console.print("[yellow]No matches.[/yellow]")
                idx.close()
                raise typer.Exit(0)
            for rank, hit in enumerate(hits, start=1):
                console.print(
                    f"[bold]{rank}.[/bold] {hit.rel_path}:{hit.start_line}-{hit.end_line}"
                )
                console.print(format_snippet(hit, cfg))
                console.print()
            idx.close()
            return

        llm = _ensure_llm_ready(cfg)
        hits, effective_query = retrieve_for_question(
            idx,
            joined,
            cfg,
            llm=llm,
            allow_clarify=not no_clarify,
            console=console,
        )
        if not hits:
            console.print("[yellow]No matches.[/yellow]")
            idx.close()
            raise typer.Exit(0)

        context = hits_to_context(hits)
        messages = build_messages(effective_query, context)

        buffer = ""
        try:
            with Live(Markdown(""), console=console, refresh_per_second=20) as live:
                for tok in llm.chat_stream(messages):
                    buffer += tok
                    live.update(Markdown(buffer))
        except LLMError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(3) from exc

        console.print("\n[bold]Sources[/bold]")
        for hit in hits:
            console.print(
                f"  [green]{hit.rel_path}[/green]"
                f":[magenta]{hit.start_line}-{hit.end_line}[/magenta]"
            )
        console.print(
            "[dim]Tip: run `documind ask` (no question) or `documind chat` "
            "to continue this thread.[/dim]"
        )

        with contextlib.suppress(Exception):
            append_turn(root, "default", joined, buffer, cfg=cfg)

        idx.close()
    except DocuMindError as exc:
        _handle_user_error(exc)


# --------------------------------------------------------------------- chat


@app.command("chat")
def cmd_chat(
    path: Path | None = typer.Option(None, "--path", "-p", help="Project root."),
    model: str | None = typer.Option(None, "--model", "-m", help="Ollama model override."),
    k: int | None = typer.Option(None, "--k", "-k", help="Snippets per question."),
    keep_alive: str | None = typer.Option(
        None,
        "--keep-alive",
        help=(
            "How long Ollama keeps the model in RAM after each reply "
            "(e.g. 5m, 0 to unload immediately). Saves idle RAM/CPU/energy."
        ),
    ),
    thread: str = typer.Option(
        "default",
        "--thread",
        "-t",
        help="Named chat thread under .documind/chats/ (default: default).",
    ),
) -> None:
    """Start an interactive chat grounded in your project.

    Threads persist under ``.documind/chats/``. Slash commands include
    ``/new``, ``/switch``, ``/threads``, ``/rename``, ``/delete``.
    """
    from .chat import run_chat

    try:
        root = _resolve_root(path)
        cfg = _make_cfg(model, k, keep_alive)
        run_chat(root, cfg, thread_name=thread)
    except DocuMindError as exc:
        _handle_user_error(exc)


# ---------------------------------------------------------------------- setup


@app.command("setup")
def cmd_setup(
    path: Path | None = typer.Option(
        None, "--path", "-p", help="Project to scan for the recommendation."
    ),
    tier: str | None = typer.Option(
        None, "--tier", help="Force a tier: tiny, small, or deep."
    ),
    model: str | None = typer.Option(
        None,
        "--model",
        "-m",
        help="Force a specific Ollama model tag (e.g. qwen2.5-coder:14b).",
    ),
    yes: bool = typer.Option(
        False, "--yes", "-y", help="Accept the recommendation without prompting."
    ),
    pull: bool | None = typer.Option(
        None,
        "--pull/--no-pull",
        help="Pull the model via Ollama. Default: ask (or skip in non-interactive shells).",
    ),
    keep_alive: str | None = typer.Option(
        None,
        "--keep-alive",
        help=(
            "Default Ollama keep_alive for ask/chat after setup "
            "(e.g. 5m). Models unload after idle to save RAM/energy."
        ),
    ),
) -> None:
    """Pick a local model for `documind ask` / `documind chat`.

    Search and index never need a model, so this command is optional.
    It saves your model preference and can (optionally) pull it via Ollama.
    Hardware RAM is used to filter which catalog models fit comfortably.
    """
    from .setup import run_setup

    root = _resolve_root(path)
    try:
        code = run_setup(
            root,
            tier=tier,
            model=model,
            yes=yes,
            pull=pull,
            keep_alive=keep_alive,
        )
    except DocuMindError as exc:
        _handle_user_error(exc)
        return
    if code != 0:
        raise typer.Exit(code)


# --------------------------------------------------------------------- models


@app.command("models")
def cmd_models() -> None:
    """List the full free/local Ollama model catalog (sizes, RAM, tradeoffs)."""
    from .models import catalog_table_rows

    table = Table(
        title="DocuMind model catalog (all free, all local)", header_style="bold"
    )
    table.add_column("Tag")
    table.add_column("Size")
    table.add_column("RAM")
    table.add_column("Speed/quality")
    table.add_column("Description")
    for tag, size, ram, tradeoff, desc in catalog_table_rows():
        table.add_row(tag, size, ram, tradeoff, desc)
    console.print(table)
    console.print(
        "[dim]Run `documind setup` to pick one that fits your machine. "
        "Override anytime with --model <tag>.[/dim]"
    )


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

    if write_config:
        p = write_default_config()
        table.add_row("config", "[green]written[/green]", str(p))
    else:
        table.add_row(
            "config", "[green]ok[/green]", f"model={cfg.model}, emb={cfg.embedding_model}"
        )

    if not ollama_installed():
        table.add_row(
            "ollama daemon",
            "[red]missing[/red]",
            f"Install: {install_hint()}",
        )
        llm = None
    else:
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

    if llm is not None and llm.model_available():
        table.add_row("model", "[green]ok[/green]", cfg.model)
    else:
        if pull and llm is not None:
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

    idx = DocuMindIndex(root, cfg)
    if idx.exists():
        chunks = len(idx.all_chunks())
        table.add_row("index", "[green]ok[/green]", f"{chunks} chunks at {idx.index_dir}")
    else:
        table.add_row("index", "[yellow]none[/yellow]", f"Run: documind index {root}")
    table.add_row(
        "workflow",
        "[dim]tip[/dim]",
        "After `git checkout` / merge, run `documind index`. "
        "Only one indexer at a time (see `.documind-index.lock`).",
    )
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
    except DocuMindError as exc:
        console.print(format_user_error(exc))
        if DEBUG:
            console.print(traceback.format_exc())
        sys.exit(exc.exit_code)
    except KeyboardInterrupt:
        console.print()
        sys.exit(130)


if __name__ == "__main__":
    main()
