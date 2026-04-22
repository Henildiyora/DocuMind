"""One-command DocuMind setup.

Responsibilities:
- Scan a target project to count files and lines of code.
- Recommend a model tier based on project size.
- Verify that Ollama is installed/reachable; if not, print clear guidance.
- Offer to pull the chosen model (or pull it directly in --yes mode).
- Persist the chosen model to the user's `~/.config/documind/config.toml`.
"""

from __future__ import annotations

import platform
import shutil
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


def _install_hint_for_ollama() -> str:
    system = platform.system()
    if system == "Darwin":
        return "brew install ollama   # or download from https://ollama.com/download"
    if system == "Linux":
        return "curl -fsSL https://ollama.com/install.sh | sh"
    return "See https://ollama.com/download"


def _render_tier_table(console: Console, recommended: str) -> None:
    table = Table(title="Available model tiers", header_style="bold")
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


def _select_spec(
    stats: ProjectStats,
    tier: str | None,
    model: str | None,
    yes: bool,
    console: Console,
) -> ModelSpec:
    """Pick a ModelSpec from explicit overrides, interactive input, or recommendation."""
    if model:
        # Custom model: synthesize a spec entry carrying the user-chosen tag.
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
    if yes:
        console.print(f"[green]Using recommended tier:[/green] {recommended}")
        return tier_info(recommended)

    choice = Prompt.ask(
        "Pick a tier",
        choices=list(TIER_ORDER),
        default=recommended,
    )
    return tier_info(choice)


def run_setup(
    root: Path,
    tier: str | None = None,
    model: str | None = None,
    yes: bool = False,
    skip_pull: bool = False,
) -> int:
    """Interactive or scripted setup flow. Returns an exit code (0 on success)."""
    console = Console()
    console.print(Panel.fit("DocuMind setup", style="bold blue"))

    # 1) Scan project
    console.print(f"Scanning [cyan]{root}[/cyan]...")
    stats = scan_project(root)
    console.print(f"Found [bold]{stats.summary}[/bold]")

    # 2) Pick model
    spec = _select_spec(stats, tier, model, yes, console)
    console.print(
        f"Selected model: [bold cyan]{spec.name}[/bold cyan] ({spec.family})"
    )

    # 3) Persist choice to user config
    cfg_path = update_user_config({"model": spec.name})
    console.print(f"Saved to [dim]{cfg_path}[/dim]")

    # 4) Verify Ollama
    cfg = load_config()
    llm = OllamaClient(cfg)

    if shutil.which("ollama") is None:
        console.print(
            Panel(
                "Ollama is not installed. Install it with:\n\n"
                f"  [bold]{_install_hint_for_ollama()}[/bold]\n\n"
                "Then run [bold]documind setup[/bold] again.",
                title="Ollama missing",
                border_style="yellow",
            )
        )
        return 2

    if not llm.ping():
        console.print(
            Panel(
                f"Ollama binary found but daemon is not reachable at "
                f"{cfg.ollama_base_url}.\n\n"
                "Start it in another terminal:\n\n"
                "  [bold]ollama serve[/bold]\n\n"
                "Then re-run [bold]documind setup[/bold].",
                title="Ollama not running",
                border_style="yellow",
            )
        )
        return 3

    console.print(f"[green]Ollama is running[/green] at {cfg.ollama_base_url}")

    # 5) Pull model if missing
    if llm.model_available(spec.name):
        console.print(f"[green]Model already pulled:[/green] {spec.name}")
    else:
        if skip_pull:
            console.print(
                f"[yellow]Model {spec.name} not pulled.[/yellow] "
                f"Run: [bold]ollama pull {spec.name}[/bold]"
            )
            return 4

        should_pull = yes or Confirm.ask(
            f"Pull [bold]{spec.name}[/bold] now? (~{spec.size_gb:.1f} GB)",
            default=True,
        )
        if not should_pull:
            console.print(
                f"[yellow]Skipped.[/yellow] Later: [bold]ollama pull {spec.name}[/bold]"
            )
            return 5

        try:
            console.print(f"Pulling [cyan]{spec.name}[/cyan]... this can take a few minutes.")
            llm.pull(spec.name)
            console.print(f"[green]Pulled[/green] {spec.name}")
        except LLMError as exc:
            console.print(f"[red]Failed to pull {spec.name}:[/red] {exc}")
            return 6

    # 6) Final status
    console.print(
        Panel(
            f"DocuMind is ready.\n\n"
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
