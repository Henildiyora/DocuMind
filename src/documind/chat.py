"""Interactive Rich-based REPL with streaming local-LLM responses."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

from .config import Config
from .errors import IndexMissing, ModelNotPulled, OllamaNotInstalled, OllamaNotRunning
from .freshness import maybe_warn_stale_index
from .index import DocuMindIndex
from .llm import LLMError, OllamaClient
from .ollama_daemon import ensure_daemon_running, ollama_installed
from .prompts import build_messages
from .query_understand import retrieve_for_question
from .search import hits_to_context
from .threads import (
    DEFAULT_THREAD,
    delete_thread,
    ensure_thread,
    list_threads,
    load_thread,
    rename_thread,
    sanitize_thread_name,
    save_thread,
)

SLASH_HELP = """
Slash commands:
  /help              show this help
  /clear             clear conversation memory for this thread
  /k N               set top-k (current: {k})
  /model M           switch LLM model for this session (current: {model})
  /new <name>        create and switch to a new thread
  /switch <name>     switch to an existing thread
  /threads           list threads with last-message preview
  /rename <old> <new> rename a thread
  /delete <name>     delete a thread
  /exit              quit
""".strip()


def run_chat(
    project_root: Path,
    cfg: Config,
    thread_name: str = DEFAULT_THREAD,
) -> None:
    """Start the interactive chat REPL with persistent named threads."""
    console = Console()
    idx = DocuMindIndex(project_root, cfg)
    if not idx.exists():
        raise IndexMissing()

    maybe_warn_stale_index(console, idx, project_root, cfg, enabled=True)

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

    try:
        current = sanitize_thread_name(thread_name)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        current = DEFAULT_THREAD

    thread = ensure_thread(project_root, current, cfg)
    session_model = cfg.model
    session_k = cfg.top_k
    history_turns = cfg.chat_history_turns

    console.print(
        Panel(
            f"DocuMind chat  |  model: [cyan]{session_model}[/cyan]  |  "
            f"thread: [magenta]{thread.name}[/magenta]  |  "
            f"project: [green]{project_root.name}[/green]\n"
            f"Type [bold]/help[/bold] for commands, [bold]/exit[/bold] to quit.\n"
            f"[dim]keep_alive={cfg.keep_alive} "
            f"(model unloads after idle to save RAM/energy)[/dim]",
            title="DocuMind",
            border_style="blue",
        )
    )

    while True:
        try:
            user = Prompt.ask("[bold cyan]you[/bold cyan]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            break
        if not user:
            continue

        if user.startswith("/"):
            parts = user[1:].split()
            cmd = parts[0].lower() if parts else ""
            if cmd in {"exit", "quit", "q"}:
                break
            if cmd in {"help", "h", "?"}:
                console.print(
                    SLASH_HELP.format(k=session_k, model=session_model)
                )
                continue
            if cmd == "clear":
                thread.messages.clear()
                save_thread(project_root, thread, cfg)
                console.print("[yellow]History cleared.[/yellow]")
                continue
            if cmd == "k" and len(parts) == 2 and parts[1].isdigit():
                session_k = max(1, int(parts[1]))
                console.print(f"[yellow]top_k -> {session_k}[/yellow]")
                continue
            if cmd == "model" and len(parts) == 2:
                session_model = parts[1]
                console.print(f"[yellow]model -> {session_model}[/yellow]")
                continue
            if cmd == "new" and len(parts) >= 2:
                try:
                    name = sanitize_thread_name(parts[1])
                except ValueError as exc:
                    console.print(f"[red]{exc}[/red]")
                    continue
                thread = ensure_thread(project_root, name, cfg)
                console.print(f"[yellow]Switched to new thread[/yellow] {name}")
                continue
            if cmd == "switch" and len(parts) >= 2:
                try:
                    name = sanitize_thread_name(parts[1])
                except ValueError as exc:
                    console.print(f"[red]{exc}[/red]")
                    continue
                thread = load_thread(project_root, name, cfg)
                console.print(
                    f"[yellow]Switched to[/yellow] {name} "
                    f"[dim]({len(thread.messages)} msgs)[/dim]"
                )
                continue
            if cmd == "threads":
                threads = list_threads(project_root, cfg)
                if not threads:
                    console.print("[dim]No threads yet.[/dim]")
                    continue
                for t in threads:
                    marker = " *" if t.name == thread.name else ""
                    console.print(
                        f"  [bold]{t.name}[/bold]{marker}  "
                        f"[dim]{t.updated_at}[/dim]  {t.preview()}"
                    )
                continue
            if cmd == "rename" and len(parts) >= 3:
                try:
                    thread = rename_thread(project_root, parts[1], parts[2], cfg)
                    console.print(
                        f"[yellow]Renamed[/yellow] {parts[1]} -> {parts[2]}"
                    )
                except (ValueError, FileNotFoundError, FileExistsError) as exc:
                    console.print(f"[red]{exc}[/red]")
                continue
            if cmd == "delete" and len(parts) >= 2:
                try:
                    name = sanitize_thread_name(parts[1])
                except ValueError as exc:
                    console.print(f"[red]{exc}[/red]")
                    continue
                if name == thread.name:
                    console.print(
                        "[red]Switch away from this thread before deleting it.[/red]"
                    )
                    continue
                delete_thread(project_root, name, cfg)
                console.print(f"[yellow]Deleted thread[/yellow] {name}")
                continue
            console.print("[red]Unknown command.[/red] Try /help.")
            continue

        prior = [
            m["content"]
            for m in thread.history_window(history_turns)
            if m.get("role") == "user"
        ]
        # Temporarily override top_k for this turn via a shallow cfg copy path:
        # retrieve_for_question reads cfg.top_k; pass session_k through search.
        from dataclasses import replace

        turn_cfg = replace(cfg, top_k=session_k)

        hits, effective_query = retrieve_for_question(
            idx,
            user,
            turn_cfg,
            llm=llm,
            allow_clarify=True,
            console=console,
            prior_user_turns=prior,
        )
        if not hits:
            console.print("[yellow]No matches found.[/yellow]")
            continue

        context = hits_to_context(hits)
        messages = build_messages(effective_query, context)
        messages = thread.history_window(history_turns) + messages

        buffer = ""
        console.print("[bold green]documind[/bold green]")
        try:
            with Live(Markdown(""), console=console, refresh_per_second=20) as live:
                for token in llm.chat_stream(messages, model=session_model):
                    buffer += token
                    live.update(Markdown(buffer))
        except LLMError as exc:
            console.print(f"[red]{exc}[/red]")
            continue

        from datetime import datetime, timezone

        now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        thread.messages.append({"role": "user", "content": user, "ts": now})
        thread.messages.append({"role": "assistant", "content": buffer, "ts": now})
        # Cap on-disk history to a generous window (memory for prompts is separate).
        max_msgs = max(history_turns * 2 * 4, 32)
        if len(thread.messages) > max_msgs:
            thread.messages = thread.messages[-max_msgs:]
        save_thread(project_root, thread, cfg)

        console.print("[bold]Sources[/bold]")
        for h in hits[:session_k]:
            console.print(
                f"  [green]{h.rel_path}[/green]"
                f":[magenta]{h.start_line}-{h.end_line}[/magenta]"
            )
