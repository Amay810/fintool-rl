"""Versioned selection predicates for training-positive trajectories."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .reward import RewardVector

POSITIVE_SELECTION_VERSION = "positive-v1"

_REQUIRED_DIMENSIONS = (
    "answer_correct",
    "grounded",
    "required_family_coverage",
    "execution_valid",
    "argument_valid",
    "temporal_valid",
    "format_valid",
)
_NUMERIC_FIELDS = (*_REQUIRED_DIMENSIONS, "efficiency", "total")
_MISSING = object()


@dataclass(frozen=True)
class PositiveSelection:
    """The auditable result of applying a positive-trajectory contract."""

    selected: bool
    version: str
    failed_conditions: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected": self.selected,
            "selection_version": self.version,
            "failed_conditions": list(self.failed_conditions),
        }


def select_positive_trajectory(
    reward: RewardVector,
    *,
    selection_version: str = POSITIVE_SELECTION_VERSION,
) -> PositiveSelection:
    """Select a trajectory for imitation data under the versioned positive contract.

    Efficiency and total reward are intentionally diagnostic only. Every correctness,
    validity, and coverage dimension must be exactly one, and hard failures fail closed.
    """
    if selection_version != POSITIVE_SELECTION_VERSION:
        raise ValueError(f"unknown positive selection version: {selection_version}")

    failed: list[str] = []
    reward_version = getattr(reward, "version", _MISSING)
    if reward_version is _MISSING:
        failed.append("missing_field:version")
    elif not isinstance(reward_version, str):
        failed.append("invalid_type:version")

    hard_failure = getattr(reward, "hard_failure", _MISSING)
    if hard_failure is _MISSING:
        failed.append("missing_field:hard_failure")
    elif hard_failure is not None and not isinstance(hard_failure, str):
        failed.append("invalid_type:hard_failure")
    elif hard_failure is not None:
        failed.append("hard_failure")

    for field in _NUMERIC_FIELDS:
        value = getattr(reward, field, _MISSING)
        if value is _MISSING:
            failed.append(f"missing_field:{field}")
        elif isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            failed.append(f"invalid_type:{field}")
        elif field in _REQUIRED_DIMENSIONS and value != 1.0:
            failed.append(field)

    return PositiveSelection(
        selected=not failed,
        version=selection_version,
        failed_conditions=tuple(failed),
    )


def is_positive_trajectory(
    reward: RewardVector,
    *,
    selection_version: str = POSITIVE_SELECTION_VERSION,
) -> bool:
    """Return only the selection bit for callers that do not need diagnostics."""
    return select_positive_trajectory(reward, selection_version=selection_version).selected
