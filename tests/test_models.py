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


def test_tier_order_is_three_small_to_deep() -> None:
    assert TIER_ORDER == ("tiny", "small", "deep")


def test_large_tier_was_dropped() -> None:
    assert all(m.tier != "large" for m in MODEL_TIERS)


def test_deprecated_tier_names_still_resolve() -> None:
    # Configs written by older DocuMind versions keep working.
    assert tier_info("medium").tier == "deep"
    assert tier_info("large").tier == "deep"


def test_tier_info_is_case_insensitive() -> None:
    assert tier_info("TINY").tier == "tiny"
    assert tier_info("Small").tier == "small"
    assert tier_info("Deep").tier == "deep"


def test_tier_info_rejects_unknown() -> None:
    with pytest.raises(ValueError):
        tier_info("XXL")


def test_default_model_is_the_tiniest() -> None:
    assert default_model() == tier_info("tiny").name


@pytest.mark.parametrize(
    ("files", "loc", "expected"),
    [
        (0,       0,         "tiny"),
        (10,      500,       "tiny"),
        (200,     50_000,    "tiny"),
        (201,     50_001,    "small"),
        (1_000,   200_000,   "small"),
        (1_500,   500_000,   "small"),
        (1_501,   500_001,   "deep"),
        (5_000,   5_000_000, "deep"),
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


def test_no_tier_exceeds_five_gb() -> None:
    # Enforces the "no giant default" rule.
    assert max(m.size_gb for m in MODEL_TIERS) <= 5.0
