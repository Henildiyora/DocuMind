"""Curated Ollama model tiers and a size-based recommender.

DocuMind ships with a deliberately small set of tiers. Users who want a
specific model can always pass `--model <tag>`; tiers exist to give newcomers
a sane default based on their project's size.

All listed models are free and run 100% locally via Ollama.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    """A recommended Ollama model entry."""

    tier: str          # "tiny" | "small" | "deep"
    name: str          # Ollama tag, e.g. "gemma3:4b"
    size_gb: float     # approximate on-disk / RAM footprint
    family: str        # short family label, e.g. "Gemma 3", "Qwen2.5 Coder"
    best_for: str      # one-line positioning

    @property
    def display(self) -> str:
        return f"{self.tier.title():<7} {self.name:<22} {self.size_gb:>4.1f} GB  {self.best_for}"


MODEL_TIERS: tuple[ModelSpec, ...] = (
    ModelSpec(
        tier="tiny",
        name="qwen2.5-coder:1.5b",
        size_gb=1.0,
        family="Qwen2.5 Coder",
        best_for="Default. Fast, free, fits on any laptop",
    ),
    ModelSpec(
        tier="small",
        name="gemma3:4b",
        size_gb=3.3,
        family="Gemma 3",
        best_for="Richer answers for mid-sized repos",
    ),
    ModelSpec(
        tier="deep",
        name="qwen2.5-coder:7b",
        size_gb=4.7,
        family="Qwen2.5 Coder",
        best_for="Deeper code reasoning on larger repos",
    ),
)


TIER_ORDER = ("tiny", "small", "deep")


def _by_tier() -> dict[str, ModelSpec]:
    return {m.tier: m for m in MODEL_TIERS}


def tier_info(tier: str) -> ModelSpec:
    """Return the ModelSpec for a tier name (case-insensitive).

    The old `medium` / `large` tier names are accepted as deprecated aliases
    so that configs written by DocuMind 2.1.0 still resolve.
    """
    key = tier.strip().lower()
    aliases = {"medium": "deep", "large": "deep"}
    key = aliases.get(key, key)
    try:
        return _by_tier()[key]
    except KeyError as exc:
        valid = ", ".join(TIER_ORDER)
        raise ValueError(f"Unknown tier {tier!r}. Choose one of: {valid}") from exc


def default_model() -> str:
    """The tag used when no recommendation has been made yet.

    The default is the smallest tier so first-time users get a quick pull.
    """
    return tier_info("tiny").name


def recommend_tier(file_count: int, total_loc: int) -> str:
    """Pick a tier based on project size.

    Thresholds lean toward smaller models because search and index never
    need an LLM at all. The tier only matters for `documind ask` / `chat`.
    """
    file_count = max(0, int(file_count))
    total_loc = max(0, int(total_loc))

    if file_count <= 200 and total_loc <= 50_000:
        return "tiny"
    if file_count <= 1_500 and total_loc <= 500_000:
        return "small"
    return "deep"


def tier_table_rows() -> list[tuple[str, str, str, str]]:
    """Return rows of (tier, model, size, best_for) for UI rendering."""
    return [
        (m.tier, m.name, f"{m.size_gb:.1f} GB", m.best_for)
        for m in MODEL_TIERS
    ]


def recommend_for_project(file_count: int, total_loc: int) -> ModelSpec:
    """Convenience: pick a tier and return the full ModelSpec."""
    return tier_info(recommend_tier(file_count, total_loc))
