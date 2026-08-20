from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from fintool_rl.reward import RewardVector
from fintool_rl.selection import (
    POSITIVE_SELECTION_VERSION,
    PositiveSelection,
    is_positive_trajectory,
    select_positive_trajectory,
)


def _reward(**overrides) -> RewardVector:
    values = {
        "execution_valid": 1.0,
        "answer_correct": 1.0,
        "argument_valid": 1.0,
        "temporal_valid": 1.0,
        "grounded": 1.0,
        "format_valid": 1.0,
        "efficiency": 1.0,
        "required_family_coverage": 1.0,
        "hard_failure": None,
        "total": 1.0,
    }
    values.update(overrides)
    return RewardVector(**values)


def test_perfect_trajectory_is_positive():
    result = select_positive_trajectory(_reward())

    assert result == PositiveSelection(selected=True, version=POSITIVE_SELECTION_VERSION)
    assert result.to_dict() == {
        "selected": True,
        "selection_version": "positive-v1",
        "failed_conditions": [],
    }


def test_correct_grounded_inefficient_trajectory_is_positive():
    result = select_positive_trajectory(_reward(efficiency=0.0, total=0.9))

    assert result.selected is True
    assert result.failed_conditions == ()
    assert is_positive_trajectory(_reward(efficiency=0.0, total=0.9)) is True


@pytest.mark.parametrize(
    "field",
    [
        "answer_correct",
        "grounded",
        "required_family_coverage",
        "execution_valid",
        "argument_valid",
        "temporal_valid",
        "format_valid",
    ],
)
def test_any_required_dimension_failure_is_not_positive(field: str):
    result = select_positive_trajectory(_reward(**{field: 0.0}))

    assert result.selected is False
    assert field in result.failed_conditions


def test_hard_failure_is_not_positive_even_with_correct_answer():
    result = select_positive_trajectory(_reward(hard_failure="execution_failure"))

    assert result.selected is False
    assert result.failed_conditions == ("hard_failure",)


def test_missing_field_fails_closed():
    reward = SimpleNamespace(
        hard_failure=None,
        answer_correct=1.0,
        grounded=1.0,
        required_family_coverage=1.0,
        execution_valid=1.0,
        argument_valid=1.0,
        temporal_valid=1.0,
        efficiency=1.0,
        total=1.0,
        version="m1-v2",
        # format_valid is intentionally absent.
    )

    result = select_positive_trajectory(reward)  # type: ignore[arg-type]

    assert result.selected is False
    assert result.failed_conditions == ("missing_field:format_valid",)


def test_wrong_field_type_fails_closed():
    result = select_positive_trajectory(replace(_reward(), grounded="1.0"))  # type: ignore[arg-type]

    assert result.selected is False
    assert result.failed_conditions == ("invalid_type:grounded",)


def test_invalid_diagnostic_field_fails_closed():
    result = select_positive_trajectory(replace(_reward(), total="1.0"))  # type: ignore[arg-type]

    assert result.selected is False
    assert result.failed_conditions == ("invalid_type:total",)


def test_unknown_selection_version_is_rejected():
    with pytest.raises(ValueError, match="unknown positive selection version"):
        select_positive_trajectory(_reward(), selection_version="positive-v2")
