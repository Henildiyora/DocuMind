"""Interactive Rich-based REPL with streaming Gemma responses."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

from .config import Config
from .index import DocuMindIndex
from .llm import LLMError, OllamaClient
from .ollama_daemon import ensure_daemon_running, install_hint
from .prompts import build_messages
from .search import hits_to_context, search

SLASH_HELP = """
Slash commands:
  /help     show this help
  /clear    clear conversation memory
  /k N      set top-k (current: {k})
  /model M  switch LLM model for this session (current: {model})
  /exit     quit
""".strip()


def run_chat(project_root: Path, cfg: Config) -> None:
    """Start the interactive chat REPL."""
    console = Console()
    idx = DocuMindIndex(project_root, cfg)
    if not idx.exists():
        console.print(
            "[red]No index found.[/red] Run [bold]documind index[/bold] first."
        )
        return

    import shutil

    if shutil.which("ollama") is None:
        console.print(
            f"[yellow]Ollama isn't installed.[/yellow] Install with:\n"
            f"  [bold]{install_hint()}[/bold]\n"
            "Tip: [bold]documind search[/bold] works without any model."
        )
        return

    status = ensure_daemon_running(cfg)
    if not status.running:
        console.print(
            "[red]Couldn't start Ollama automatically.[/red] "
            "Try [bold]ollama serve[/bold] in another terminal."
        )
        return
    if status.how != "already":
        console.print(f"[dim]Started Ollama via {status.how}.[/dim]")

    llm = OllamaClient(cfg)
    if not llm.model_available():
        console.print(
            f"[yellow]Model {cfg.model} is not pulled.[/yellow] "
            f"Run [bold]documind setup --pull[/bold] first."
        )
        return

    history: list[dict] = []
    session_model = cfg.model
    session_k = cfg.top_k

    console.print(
        Panel(
            f"DocuMind chat  |  model: [cyan]{session_model}[/cyan]  |  "
            f"project: [green]{project_root.name}[/green]\n"
            f"Type [bold]/help[/bold] for commands, [bold]/exit[/bold] to quit.",
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
                console.print(SLASH_HELP.format(k=session_k, model=session_model))
                continue
            if cmd == "clear":
                history.clear()
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
            console.print("[red]Unknown command.[/red] Try /help.")
            continue

        hits = search(idx, user, cfg, top_k=session_k)
        if not hits:
            console.print("[yellow]No matches found.[/yellow]")
            continue

        context = hits_to_context(hits)
        messages = build_messages(user, context)
        # Keep a short rolling history for follow-ups
        messages = history + messages
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

        # Light summarized memory: append last exchange
        history.append({"role": "user", "content": user})
        history.append({"role": "assistant", "content": buffer})
        if len(history) > 8:
            history = history[-8:]

        # Footer: show which files were used
        refs = ", ".join(
            f"{h.rel_path}:{h.start_line}-{h.end_line}" for h in hits[: session_k]
        )
        console.print(f"[dim]sources: {refs}[/dim]")
