"""One-command DocuMind setup.

Design principle: never block the user. Scanning and saving a model
preference always succeeds; pulling the model via Ollama is a best-effort
add-on. Search and index work regardless.

Responsibilities:
- Scan a target project to count files and lines of code.
- Recommend a model based on hardware RAM (and project size as a tie-break).
- Save the chosen model to `~/.config/documind/config.toml`.
- If requested, auto-start Ollama (without forcing a second terminal) and
  pull the chosen model.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table

from .chunker import iter_source_files
from .config import Config, load_config, update_user_config, user_config_path
from .errors import ModelPullFailed, OllamaNotInstalled, OllamaNotRunning
from .llm import LLMError, OllamaClient
from .models import (
    MODEL_TIERS,
    TIER_ORDER,
    ModelSpec,
    detect_gpu_hint,
    detect_system_ram_gb,
    filter_catalog_for_ram,
    recommend_for_hardware,
    recommend_tier,
    tier_info,
)
from .ollama_daemon import ensure_daemon_running, install_hint, ollama_installed


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


def _render_catalog_table(
    console: Console,
    candidates: list[ModelSpec],
    recommended: ModelSpec,
    *,
    ram_gb: float | None,
    gpu: str | None,
) -> None:
    title = "Models that fit this machine  (all free, all local)"
    if ram_gb is not None:
        title += f"  |  ~{ram_gb:.0f} GB RAM"
    if gpu:
        title += f"  |  {gpu}"
    table = Table(title=title, header_style="bold")
    table.add_column("#")
    table.add_column("Model")
    table.add_column("Size")
    table.add_column("RAM")
    table.add_column("Tradeoff")
    table.add_column("Best for")

    for i, m in enumerate(candidates, start=1):
        marker = " [green](recommended)[/green]" if m.name == recommended.name else ""
        table.add_row(
            str(i),
            f"{m.name}{marker}",
            f"{m.size_gb:.1f} GB",
            f"~{m.ram_gb:.0f} GB",
            m.tradeoff,
            m.best_for,
        )
    console.print(table)
    console.print(
        "[dim]Tip: any Ollama tag works via `--model <tag>`. "
        "List the full catalog with `documind models`.[/dim]"
    )


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
            ram_gb=0.0,
            family="custom",
            best_for="User-specified model",
            tradeoff="balanced",
        )

    if tier:
        return tier_info(tier)

    ram_gb = detect_system_ram_gb()
    gpu = detect_gpu_hint()
    recommended = recommend_for_hardware(
        ram_gb, stats.file_count, stats.total_loc
    )
    candidates = filter_catalog_for_ram(ram_gb)

    if ram_gb is None:
        # Fall back to classic tiny/small/deep table when RAM unknown.
        classic = recommend_tier(stats.file_count, stats.total_loc)
        _render_tier_table(console, classic)
        if yes or not sys.stdin.isatty():
            console.print(f"[green]Using recommended tier:[/green] {classic}")
            return tier_info(classic)
        choice = Prompt.ask(
            "Pick a tier",
            choices=list(TIER_ORDER),
            default=classic,
        )
        return tier_info(choice)

    _render_catalog_table(
        console, candidates, recommended, ram_gb=ram_gb, gpu=gpu
    )
    if yes or not sys.stdin.isatty():
        console.print(f"[green]Using recommended model:[/green] {recommended.name}")
        return recommended

    # Let the user pick by number or by exact tag.
    default = recommended.name
    choice = Prompt.ask(
        "Pick a model tag (or number)",
        default=default,
    )
    choice = choice.strip()
    if choice.isdigit():
        idx = int(choice) - 1
        if 0 <= idx < len(candidates):
            return candidates[idx]
    for m in candidates:
        if m.name == choice:
            return m
    # Allow typing any tag even if filtered out.
    return ModelSpec(
        tier="custom",
        name=choice,
        size_gb=0.0,
        ram_gb=0.0,
        family="custom",
        best_for="User-specified model",
        tradeoff="balanced",
    )


def _should_pull(pull: bool | None, yes: bool, console: Console, spec: ModelSpec) -> bool:
    """Resolve the --pull/--no-pull tri-state into a boolean."""
    if pull is True:
        return True
    if pull is False:
        return False
    if yes:
        return True
    if not sys.stdin.isatty():
        return False
    size = f"~{spec.size_gb:.1f} GB, " if spec.size_gb else ""
    return Confirm.ask(
        f"Pull [bold]{spec.name}[/bold] now? ({size}free, local)",
        default=True,
    )


def run_setup(
    root: Path,
    tier: str | None = None,
    model: str | None = None,
    yes: bool = False,
    pull: bool | None = None,
    keep_alive: str | None = None,
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

    console.print(f"Scanning [cyan]{root}[/cyan]...")
    stats = scan_project(root)
    console.print(f"Found [bold]{stats.summary}[/bold]")

    ram = detect_system_ram_gb()
    if ram is not None:
        console.print(f"Detected [bold]~{ram:.0f} GB[/bold] system RAM")
    gpu = detect_gpu_hint()
    if gpu:
        console.print(f"GPU hint: [bold]{gpu}[/bold]")

    try:
        spec = _select_spec(stats, tier, model, yes, console)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        return 1
    console.print(
        f"Selected model: [bold cyan]{spec.name}[/bold cyan] ({spec.family})"
    )

    updates: dict = {"model": spec.name, "setup_done": True}
    if keep_alive is not None:
        updates["keep_alive"] = keep_alive
    cfg_path = update_user_config(updates)
    console.print(f"Saved to [dim]{cfg_path}[/dim]")
    if keep_alive is not None:
        console.print(
            f"[dim]keep_alive={keep_alive} "
            f"(model unloads after idle to save RAM/CPU/energy)[/dim]"
        )
    else:
        console.print(
            "[dim]Default keep_alive=5m — Ollama unloads the model after idle "
            "so it does not burn RAM forever. Override with --keep-alive.[/dim]"
        )

    console.print(
        "[green]Ready.[/green] Search and index already work. "
        "The rest is only needed for [bold]documind ask[/bold] / [bold]documind chat[/bold]."
    )

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

    if not ollama_installed():
        if pull is True:
            raise OllamaNotInstalled()
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
        return 0

    cfg = load_config()
    status = ensure_daemon_running(cfg)
    if not status.running:
        if pull is True:
            raise OllamaNotRunning()
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
        return 0
    if status.how != "already":
        console.print(f"[green]Started Ollama[/green] via {status.how}.")

    llm = OllamaClient(cfg)
    if llm.model_available(spec.name):
        console.print(f"[green]Model already pulled:[/green] {spec.name}")
    else:
        try:
            console.print(f"Pulling [cyan]{spec.name}[/cyan]... (this can take a few minutes)")
            llm.pull(spec.name)
            console.print(f"[green]Pulled[/green] {spec.name}")
        except LLMError as exc:
            if pull is True:
                raise ModelPullFailed(spec.name, str(exc)) from exc
            console.print(f"[red]Failed to pull {spec.name}:[/red] {exc}")
            return 2

    console.print(
        Panel(
            "DocuMind is fully set up.\n\n"
            f"Model: [bold]{spec.name}[/bold]\n"
            f"Config: [dim]{user_config_path()}[/dim]\n"
            f"keep_alive: [bold]{load_config().keep_alive}[/bold] "
            "(unloads after idle — lower RAM/energy use)\n\n"
            "Next steps:\n"
            "  cd /path/to/any/project\n"
            "  documind index\n"
            "  documind search \"your query\"\n"
            "  documind ask why does X work",
            title="All set",
            border_style="green",
        )
    )
    return 0
