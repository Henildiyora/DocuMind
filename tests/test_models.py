"""Tests for the curated model tiers and project-size recommender."""

from __future__ import annotations

import pytest

from documind.models import (
    MODEL_TIERS,
    TIER_ORDER,
    default_model,
    recommend_for_project,
    recommend_tier,
    tier_info,
)


def test_tier_order_matches_registry_shape() -> None:
    registry = {m.tier for m in MODEL_TIERS}
    assert registry == set(TIER_ORDER)


def test_tier_info_is_case_insensitive() -> None:
    assert tier_info("TINY").tier == "tiny"
    assert tier_info("Small").tier == "small"


def test_tier_info_rejects_unknown() -> None:
    with pytest.raises(ValueError):
        tier_info("XXL")


def test_default_model_points_to_small_tier() -> None:
    assert default_model() == tier_info("small").name


@pytest.mark.parametrize(
    ("files", "loc", "expected"),
    [
        (0,       0,         "tiny"),
        (10,      500,       "tiny"),
        (30,      5_000,     "tiny"),
        (31,      5_001,     "small"),
        (100,     50_000,    "small"),
        (300,     100_000,   "small"),
        (301,     100_001,   "medium"),
        (500,     250_000,   "medium"),
        (1_500,   1_000_000, "medium"),
        (1_501,   1_000_001, "large"),
        (5_000,   5_000_000, "large"),
    ],
)
def test_recommend_tier_thresholds(files: int, loc: int, expected: str) -> None:
    assert recommend_tier(files, loc) == expected


def test_recommend_tier_handles_negatives() -> None:
    assert recommend_tier(-10, -500) == "tiny"


def test_recommend_for_project_returns_spec() -> None:
    spec = recommend_for_project(10, 100)
    assert spec.tier == "tiny"
    assert spec.name.startswith("qwen2.5-coder")
