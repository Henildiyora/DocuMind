"""Curated Ollama model tiers and a size-based recommender.

DocuMind recommends a model tier based on the target project's size
(number of source files and lines of code). Users can always override.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    """A recommended Ollama model entry."""

    tier: str          # "tiny" | "small" | "medium" | "large"
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
        best_for="Tiny projects, fastest answers",
    ),
    ModelSpec(
        tier="small",
        name="gemma3:4b",
        size_gb=3.3,
        family="Gemma 3",
        best_for="Balanced default for most repos",
    ),
    ModelSpec(
        tier="medium",
        name="qwen2.5-coder:7b",
        size_gb=4.7,
        family="Qwen2.5 Coder",
        best_for="Larger codebases, better reasoning",
    ),
    ModelSpec(
        tier="large",
        name="qwen2.5-coder:14b",
        size_gb=9.0,
        family="Qwen2.5 Coder",
        best_for="Very large repos, deep explanations",
    ),
)


TIER_ORDER = ("tiny", "small", "medium", "large")


def _by_tier() -> dict[str, ModelSpec]:
    return {m.tier: m for m in MODEL_TIERS}


def tier_info(tier: str) -> ModelSpec:
    """Return the ModelSpec for a tier name (case-insensitive)."""
    key = tier.strip().lower()
    try:
        return _by_tier()[key]
    except KeyError as exc:
        valid = ", ".join(TIER_ORDER)
        raise ValueError(f"Unknown tier {tier!r}. Choose one of: {valid}") from exc


def default_model() -> str:
    """The tag used when no recommendation has been made yet."""
    return tier_info("small").name


def recommend_tier(file_count: int, total_loc: int) -> str:
    """Pick a tier based on project size.

    Thresholds are tuned for a balance of speed and quality on a CPU / Apple
    Silicon machine. `total_loc` dominates for very large repos; `file_count`
    is a useful proxy for small ones.
    """
    file_count = max(0, int(file_count))
    total_loc = max(0, int(total_loc))

    if file_count <= 30 and total_loc <= 5_000:
        return "tiny"
    if file_count <= 300 and total_loc <= 100_000:
        return "small"
    if file_count <= 1_500 and total_loc <= 1_000_000:
        return "medium"
    return "large"


def tier_table_rows() -> list[tuple[str, str, str, str]]:
    """Return rows of (tier, model, size, best_for) for UI rendering."""
    return [
        (m.tier, m.name, f"{m.size_gb:.1f} GB", m.best_for)
        for m in MODEL_TIERS
    ]


def recommend_for_project(file_count: int, total_loc: int) -> ModelSpec:
    """Convenience: pick a tier and return the full ModelSpec."""
    return tier_info(recommend_tier(file_count, total_loc))
