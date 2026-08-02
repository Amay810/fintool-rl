"""Deterministic failure taxonomy for financial tool-agent baselines."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any

from .contracts import TaskSpec, ToolCall, Trajectory
from .reward import RewardVector

TAXONOMY_VERSION = "m1-baseline-v1"

# Ordered from most specific operational failure to answer-quality failures.
PRIMARY_LABELS = (
    "success",
    "model_call_error",
    "invalid_answer_format",
    "invalid_arguments",
    "temporal_violation",
    "no_tool_use",
    "max_steps",
    "tool_lookup_miss",
    "execution_failure",
    "wrong_unit",
    "correct_ungrounded",
    "wrong_value",
    "soft_failure",
)


@dataclass(frozen=True)
class FailureLabel:
    primary: str
    secondary: tuple[str, ...]
    evidence: dict[str, Any]
    taxonomy_version: str = TAXONOMY_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _tool_errors(calls: list[ToolCall]) -> list[str]:
    errors: list[str] = []
    for call in calls:
        if call.result.get("ok"):
            continue
        error = call.result.get("error") or call.error or "unknown"
        errors.append(str(error))
    return errors


def _lookup_miss(errors: list[str]) -> bool:
    markers = {
        "fact_not_available_at_cutoff",
        "company_not_found",
        "periods_not_found",
        "price_not_found",
        "price_series_not_found",
        "index_level_not_found",
        "trading_days_not_found",
    }
    return any(error in markers for error in errors)


def classify_failure(
    task: TaskSpec,
    trajectory: Trajectory,
    reward: RewardVector,
) -> FailureLabel:
    """Assign one primary failure label and optional secondary tags."""
    secondary: list[str] = []
    evidence: dict[str, Any] = {
        "terminal_reason": trajectory.terminal_reason,
        "hard_failure": reward.hard_failure,
        "n_tool_calls": len(trajectory.tool_calls),
        "template_family": task.template_family,
        "difficulty": task.difficulty,
        "split": task.split,
    }
    errors = _tool_errors(trajectory.tool_calls)
    if errors:
        evidence["tool_errors"] = errors[:8]

    if reward.hard_failure is None and reward.total == 1.0 and reward.answer_correct == 1.0:
        return FailureLabel(primary="success", secondary=tuple(secondary), evidence=evidence)

    if reward.hard_failure == "model_call_error" or trajectory.terminal_reason == "model_call_error":
        return FailureLabel(primary="model_call_error", secondary=tuple(secondary), evidence=evidence)

    if reward.hard_failure == "invalid_answer_format" or trajectory.terminal_reason == "invalid_action":
        return FailureLabel(primary="invalid_answer_format", secondary=tuple(secondary), evidence=evidence)

    if reward.hard_failure == "invalid_arguments":
        return FailureLabel(primary="invalid_arguments", secondary=tuple(secondary), evidence=evidence)

    if reward.hard_failure == "temporal_violation":
        return FailureLabel(primary="temporal_violation", secondary=tuple(secondary), evidence=evidence)

    if trajectory.terminal_reason == "max_steps":
        return FailureLabel(primary="max_steps", secondary=tuple(secondary), evidence=evidence)

    if not trajectory.tool_calls:
        return FailureLabel(primary="no_tool_use", secondary=tuple(secondary), evidence=evidence)

    if reward.hard_failure == "execution_failure":
        if _lookup_miss(errors):
            return FailureLabel(primary="tool_lookup_miss", secondary=tuple(secondary), evidence=evidence)
        return FailureLabel(primary="execution_failure", secondary=tuple(secondary), evidence=evidence)

    answer = trajectory.final_answer or {}
    expected_unit = task.answer.get("unit")
    if reward.format_valid and answer.get("unit") != expected_unit:
        secondary.append("unit_mismatch")
        return FailureLabel(primary="wrong_unit", secondary=tuple(secondary), evidence=evidence)

    if reward.answer_correct == 1.0 and reward.grounded == 0.0:
        return FailureLabel(primary="correct_ungrounded", secondary=tuple(secondary), evidence=evidence)

    if reward.format_valid and reward.answer_correct == 0.0:
        if reward.required_family_coverage < 1.0:
            secondary.append("missing_required_families")
        if reward.efficiency < 1.0:
            secondary.append("inefficient")
        return FailureLabel(primary="wrong_value", secondary=tuple(secondary), evidence=evidence)

    if reward.efficiency < 1.0:
        secondary.append("inefficient")
    if reward.required_family_coverage < 1.0:
        secondary.append("missing_required_families")
    if reward.grounded == 0.0:
        secondary.append("ungrounded")
    return FailureLabel(primary="soft_failure", secondary=tuple(secondary), evidence=evidence)


def summarize_labels(labels: list[FailureLabel]) -> dict[str, Any]:
    primary = Counter(label.primary for label in labels)
    secondary = Counter(tag for label in labels for tag in label.secondary)
    return {
        "taxonomy_version": TAXONOMY_VERSION,
        "n": len(labels),
        "primary": dict(sorted(primary.items())),
        "secondary": dict(sorted(secondary.items())),
        "success_rate": primary.get("success", 0) / max(1, len(labels)),
    }
