"""Curated Ollama model catalog and hardware-aware recommender.

All listed models are free and run 100% locally via Ollama.
"""

from __future__ import annotations

import platform
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelSpec:
    """A recommended Ollama model entry."""

    tier: str  # "tiny" | "small" | "deep" | "custom" | catalog slug
    name: str  # Ollama tag, e.g. "gemma3:4b"
    size_gb: float  # approximate on-disk footprint
    ram_gb: float  # approximate RAM needed while loaded
    family: str  # short family label
    best_for: str  # one-line plain-English positioning
    tradeoff: str = "balanced"  # "fast" | "balanced" | "quality"

    @property
    def display(self) -> str:
        return (
            f"{self.tier:<7} {self.name:<22} {self.size_gb:>4.1f} GB  "
            f"~{self.ram_gb:.0f}G RAM  {self.best_for}"
        )


# Full catalog shown by `documind models` and filtered by `setup`.
MODEL_CATALOG: tuple[ModelSpec, ...] = (
    ModelSpec(
        tier="tiny",
        name="qwen2.5-coder:0.5b",
        size_gb=0.4,
        ram_gb=1.0,
        family="Qwen2.5 Coder",
        best_for="Smallest, near-instant answers, best on any laptop",
        tradeoff="fast",
    ),
    ModelSpec(
        tier="tiny",
        name="qwen2.5-coder:1.5b",
        size_gb=1.0,
        ram_gb=2.0,
        family="Qwen2.5 Coder",
        best_for="Fast coding answers that still fit tiny machines",
        tradeoff="fast",
    ),
    ModelSpec(
        tier="tiny",
        name="llama3.2:1b",
        size_gb=1.3,
        ram_gb=2.0,
        family="Llama 3.2",
        best_for="Tiny general chat; quick and light",
        tradeoff="fast",
    ),
    ModelSpec(
        tier="tiny",
        name="phi3.5:3.8b",
        size_gb=2.2,
        ram_gb=4.0,
        family="Phi 3.5",
        best_for="Compact reasoning from Microsoft Phi",
        tradeoff="balanced",
    ),
    ModelSpec(
        tier="small",
        name="gemma3:4b",
        size_gb=3.3,
        ram_gb=6.0,
        family="Gemma 3",
        best_for="Richer answers for mid-sized repos",
        tradeoff="balanced",
    ),
    ModelSpec(
        tier="small",
        name="llama3.2:3b",
        size_gb=2.0,
        ram_gb=4.0,
        family="Llama 3.2",
        best_for="Solid general Q&A without a big footprint",
        tradeoff="balanced",
    ),
    ModelSpec(
        tier="small",
        name="qwen2.5-coder:3b",
        size_gb=1.9,
        ram_gb=4.0,
        family="Qwen2.5 Coder",
        best_for="Better code detail than 1.5b, still laptop-friendly",
        tradeoff="balanced",
    ),
    ModelSpec(
        tier="deep",
        name="qwen2.5-coder:7b",
        size_gb=4.7,
        ram_gb=8.0,
        family="Qwen2.5 Coder",
        best_for="Deeper code reasoning on larger repos",
        tradeoff="quality",
    ),
    ModelSpec(
        tier="deep",
        name="gemma3:12b",
        size_gb=8.1,
        ram_gb=12.0,
        family="Gemma 3",
        best_for="Higher-quality synthesis when you have spare RAM",
        tradeoff="quality",
    ),
    ModelSpec(
        tier="deep",
        name="llama3.1:8b",
        size_gb=4.7,
        ram_gb=10.0,
        family="Llama 3.1",
        best_for="Strong general answers if your machine has 16GB+",
        tradeoff="quality",
    ),
)


# Backward-compat: one representative per classic tier for `--tier`.
MODEL_TIERS: tuple[ModelSpec, ...] = (
    next(m for m in MODEL_CATALOG if m.name == "qwen2.5-coder:1.5b"),
    next(m for m in MODEL_CATALOG if m.name == "gemma3:4b"),
    next(m for m in MODEL_CATALOG if m.name == "qwen2.5-coder:7b"),
)


TIER_ORDER = ("tiny", "small", "deep")

# Prefer higher tradeoff quality when RAM allows.
_TRADEOFF_RANK = {"fast": 0, "balanced": 1, "quality": 2}


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
    """The tag used when no recommendation has been made yet."""
    return tier_info("tiny").name


def recommend_tier(file_count: int, total_loc: int) -> str:
    """Pick a classic tier based on project size (fallback without RAM info)."""
    file_count = max(0, int(file_count))
    total_loc = max(0, int(total_loc))

    if file_count <= 200 and total_loc <= 50_000:
        return "tiny"
    if file_count <= 1_500 and total_loc <= 500_000:
        return "small"
    return "deep"


def tier_table_rows() -> list[tuple[str, str, str, str]]:
    """Return rows of (tier, model, size, best_for) for legacy UI rendering."""
    return [
        (m.tier, m.name, f"{m.size_gb:.1f} GB", m.best_for)
        for m in MODEL_TIERS
    ]


def catalog_table_rows() -> list[tuple[str, str, str, str, str]]:
    """Rows for ``documind models``: tag, size, ram, tradeoff, description."""
    return [
        (
            m.name,
            f"{m.size_gb:.1f} GB",
            f"~{m.ram_gb:.0f} GB",
            m.tradeoff,
            m.best_for,
        )
        for m in MODEL_CATALOG
    ]


def detect_system_ram_gb() -> float | None:
    """Best-effort total system RAM in GB (stdlib only)."""
    system = platform.system()
    try:
        if system == "Darwin":
            out = subprocess.check_output(
                ["sysctl", "-n", "hw.memsize"], text=True, timeout=3
            ).strip()
            return int(out) / (1024**3)
        if system == "Linux":
            text = Path("/proc/meminfo").read_text(encoding="utf-8")
            for line in text.splitlines():
                if line.startswith("MemTotal:"):
                    parts = line.split()
                    # value is in kB
                    return int(parts[1]) / (1024**2)
    except Exception:
        return None
    return None


def detect_gpu_hint() -> str | None:
    """Return a short GPU hint string if easily detectable, else None."""
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                text=True,
                timeout=3,
            ).strip()
            if out:
                return out.splitlines()[0].strip()
        except Exception:
            return "NVIDIA GPU"
    if platform.system() == "Darwin":
        return "Apple Silicon / Metal"
    return None


def filter_catalog_for_ram(
    available_ram_gb: float | None,
    *,
    headroom: float = 0.6,
) -> list[ModelSpec]:
    """Keep catalog entries that fit in ``available_ram * headroom``."""
    if available_ram_gb is None or available_ram_gb <= 0:
        return list(MODEL_CATALOG)
    budget = available_ram_gb * headroom
    fitted = [m for m in MODEL_CATALOG if m.ram_gb <= budget]
    return fitted or list(MODEL_CATALOG[:3])  # always show at least the tiniest


def recommend_for_hardware(
    available_ram_gb: float | None,
    file_count: int = 0,
    total_loc: int = 0,
) -> ModelSpec:
    """Pick the best-quality catalog model that comfortably fits RAM.

    Project size only breaks ties toward slightly smaller models on tiny repos.
    """
    candidates = filter_catalog_for_ram(available_ram_gb)
    # Prefer quality, then larger models within that band.
    ranked = sorted(
        candidates,
        key=lambda m: (_TRADEOFF_RANK.get(m.tradeoff, 1), m.ram_gb, m.size_gb),
        reverse=True,
    )
    if not ranked:
        return MODEL_TIERS[0]

    # On very small projects, nudge one step toward faster if possible.
    if file_count <= 200 and total_loc <= 50_000 and len(ranked) > 1:
        # Take the second-best if the top is quality and next is balanced/fast.
        top = ranked[0]
        alt = ranked[1]
        if _TRADEOFF_RANK.get(top.tradeoff, 1) > _TRADEOFF_RANK.get(alt.tradeoff, 1):
            return alt
    return ranked[0]


def recommend_for_project(file_count: int, total_loc: int) -> ModelSpec:
    """Convenience: classic tier pick without RAM detection."""
    return tier_info(recommend_tier(file_count, total_loc))


def find_catalog_entry(tag: str) -> ModelSpec | None:
    """Look up a catalog entry by exact Ollama tag."""
    tag = tag.strip()
    for m in MODEL_CATALOG:
        if m.name == tag:
            return m
    return None


def parse_size_token(text: str) -> float | None:
    """Parse a rough size like '7b' from a model tag (helper for tests)."""
    m = re.search(r"(\d+(?:\.\d+)?)\s*[bB]\b", text)
    if not m:
        return None
    return float(m.group(1))
