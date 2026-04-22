"""One-command DocuMind setup.

Design principle: never block the user. Scanning and saving a model
preference always succeeds; pulling the model via Ollama is a best-effort
add-on. Search and index work regardless.

Responsibilities:
- Scan a target project to count files and lines of code.
- Recommend a model tier based on project size.
- Save the chosen model to `~/.config/documind/config.toml`.
- If requested, auto-start Ollama (without forcing a second terminal) and
  pull the chosen model.
"""

from __future__ import annotations

import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table

from .chunker import iter_source_files
from .config import Config, load_config, update_user_config, user_config_path
from .llm import LLMError, OllamaClient
from .models import (
    MODEL_TIERS,
    TIER_ORDER,
    ModelSpec,
    recommend_tier,
    tier_info,
)
from .ollama_daemon import ensure_daemon_running, install_hint


@dataclass
class ProjectStats:
    """Lightweight size summary used to pick a model tier."""

    file_count: int
    total_loc: int

    @property
    def summary(self) -> str:
        return f"{self.file_count} files, {self.total_loc:,} lines"


def scan_project(root: Path, cfg: Config | None = None) -> ProjectStats:
    """Count supported files and their total line count under `root`."""
    cfg = cfg or load_config()
    files = 0
    total_loc = 0
    for path in iter_source_files(root.resolve(), cfg.max_file_bytes):
        files += 1
        try:
            with path.open("rb") as f:
                total_loc += sum(1 for _ in f)
        except OSError:
            continue
    return ProjectStats(file_count=files, total_loc=total_loc)


def _render_tier_table(console: Console, recommended: str) -> None:
    table = Table(title="Available model tiers  (all free, all local)", header_style="bold")
    table.add_column("Tier")
    table.add_column("Model")
    table.add_column("Size")
    table.add_column("Best for")

    for m in MODEL_TIERS:
        marker = " [green](recommended)[/green]" if m.tier == recommended else ""
        table.add_row(
            f"{m.tier}{marker}",
            m.name,
            f"{m.size_gb:.1f} GB",
            m.best_for,
        )
    console.print(table)
    console.print(
        "[dim]Tip: any Ollama tag works via `--model <tag>` "
        "(e.g. --model qwen2.5-coder:14b).[/dim]"
    )


def _select_spec(
    stats: ProjectStats,
    tier: str | None,
    model: str | None,
    yes: bool,
    console: Console,
) -> ModelSpec:
    """Pick a ModelSpec from explicit overrides, interactive input, or recommendation."""
    if model:
        return ModelSpec(
            tier="custom",
            name=model,
            size_gb=0.0,
            family="custom",
            best_for="User-specified model",
        )

    recommended = recommend_tier(stats.file_count, stats.total_loc)

    if tier:
        return tier_info(tier)

    _render_tier_table(console, recommended)
    if yes or not sys.stdin.isatty():
        console.print(f"[green]Using recommended tier:[/green] {recommended}")
        return tier_info(recommended)

    choice = Prompt.ask(
        "Pick a tier",
        choices=list(TIER_ORDER),
        default=recommended,
    )
    return tier_info(choice)


def _should_pull(pull: bool | None, yes: bool, console: Console, spec: ModelSpec) -> bool:
    """Resolve the --pull/--no-pull tri-state into a boolean."""
    if pull is True:
        return True
    if pull is False:
        return False
    if yes:
        return True
    if not sys.stdin.isatty():
        # Non-interactive default: don't burn bandwidth unless asked.
        return False
    return Confirm.ask(
        f"Pull [bold]{spec.name}[/bold] now? (~{spec.size_gb:.1f} GB, free, local)",
        default=True,
    )


def run_setup(
    root: Path,
    tier: str | None = None,
    model: str | None = None,
    yes: bool = False,
    pull: bool | None = None,
) -> int:
    """Interactive or scripted setup flow. Returns an exit code.

    Exit codes:
        0 - preference saved (optionally, model pulled)
        1 - invalid tier/model argument
        2 - pull was explicitly requested but failed
    """
    console = Console()
    console.print(
        Panel.fit(
            "DocuMind setup\n[dim]100% free, 100% local. No API keys.[/dim]",
            style="bold blue",
        )
    )

    # 1) Scan project
    console.print(f"Scanning [cyan]{root}[/cyan]...")
    stats = scan_project(root)
    console.print(f"Found [bold]{stats.summary}[/bold]")

    # 2) Pick model
    try:
        spec = _select_spec(stats, tier, model, yes, console)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        return 1
    console.print(
        f"Selected model: [bold cyan]{spec.name}[/bold cyan] ({spec.family})"
    )

    # 3) Persist choice (always succeeds, independent of Ollama)
    cfg_path = update_user_config({"model": spec.name})
    console.print(f"Saved to [dim]{cfg_path}[/dim]")

    console.print(
        "[green]Ready.[/green] Search and index already work. "
        "The rest is only needed for [bold]documind ask[/bold] / [bold]documind chat[/bold]."
    )

    # 4) Decide whether to pull
    want_pull = _should_pull(pull, yes, console, spec)
    if not want_pull:
        console.print(
            Panel(
                "Skipping model download.\n\n"
                "You can always pull later with:\n"
                f"  [bold]ollama pull {spec.name}[/bold]\n"
                "or re-run [bold]documind setup --pull[/bold].",
                title="All set",
                border_style="green",
            )
        )
        return 0

    # 5) Make Ollama available (install check + auto-start)
    if shutil.which("ollama") is None:
        console.print(
            Panel(
                "Ollama is not installed. Install it with:\n\n"
                f"  [bold]{install_hint()}[/bold]\n\n"
                "Then re-run [bold]documind setup --pull[/bold]. "
                "(Search and index don't need Ollama at all.)",
                title="Ollama missing",
                border_style="yellow",
            )
        )
        return 0 if pull is None else 2

    cfg = load_config()
    status = ensure_daemon_running(cfg)
    if not status.running:
        console.print(
            Panel(
                "Ollama binary found but the daemon couldn't be started automatically.\n\n"
                "Try in another terminal:\n\n"
                "  [bold]ollama serve[/bold]\n\n"
                "Then re-run [bold]documind setup --pull[/bold].",
                title="Ollama not running",
                border_style="yellow",
            )
        )
        return 0 if pull is None else 2
    if status.how != "already":
        console.print(f"[green]Started Ollama[/green] via {status.how}.")

    # 6) Pull the model
    llm = OllamaClient(cfg)
    if llm.model_available(spec.name):
        console.print(f"[green]Model already pulled:[/green] {spec.name}")
    else:
        try:
            console.print(f"Pulling [cyan]{spec.name}[/cyan]... (this can take a few minutes)")
            llm.pull(spec.name)
            console.print(f"[green]Pulled[/green] {spec.name}")
        except LLMError as exc:
            console.print(f"[red]Failed to pull {spec.name}:[/red] {exc}")
            return 2

    # 7) Done
    console.print(
        Panel(
            "DocuMind is fully set up.\n\n"
            f"Model: [bold]{spec.name}[/bold]\n"
            f"Config: [dim]{user_config_path()}[/dim]\n\n"
            "Next steps:\n"
            "  cd /path/to/any/project\n"
            "  documind index\n"
            "  documind search \"your query\"\n"
            "  documind ask \"how does X work?\"",
            title="All set",
            border_style="green",
        )
    )
    return 0
