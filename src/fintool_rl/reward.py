"""Decomposable, replayable reward for financial tool trajectories."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from .contracts import TaskSpec, Trajectory
from .schema import TOOL_FAMILY_BY_NAME

REWARD_VERSION = "m1-v3"
EXPLORATION_ALLOWANCE = 1


@dataclass(frozen=True)
class RewardVector:
    execution_valid: float
    answer_correct: float
    argument_valid: float
    temporal_valid: float
    grounded: float
    format_valid: float
    efficiency: float
    required_family_coverage: float
    hard_failure: str | None
    total: float
    version: str = REWARD_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _answer_format(answer: Any) -> bool:
    if not isinstance(answer, dict):
        return False
    if set(answer) != {"value", "unit", "observation_ids"}:
        return False
    if isinstance(answer["value"], bool) or not isinstance(answer["value"], (int, float)):
        return False
    return isinstance(answer["unit"], str) and isinstance(answer["observation_ids"], list) and all(
        isinstance(item, str) for item in answer["observation_ids"]
    )


def grade_trajectory(task: TaskSpec, trajectory: Trajectory) -> RewardVector:
    calls = trajectory.tool_calls
    execution_valid = float(bool(calls) and all(call.result.get("ok") for call in calls))
    # An empty trace has no malformed call; it fails execution separately.
    argument_valid = float(all(call.argument_valid for call in calls))
    format_valid = float(_answer_format(trajectory.final_answer))

    temporal_ok = True
    known_observations: dict[str, dict[str, Any]] = {}
    for call in calls:
        provenance = call.result.get("provenance") or {}
        observation_id = provenance.get("observation_id")
        if observation_id:
            known_observations[observation_id] = call.result
        if provenance.get("as_of_time", task.as_of_time) > task.as_of_time:
            temporal_ok = False
        if call.result.get("error") in {"future_data_requested", "as_of_time_after_snapshot"}:
            temporal_ok = False
    temporal_valid = float(temporal_ok)

    answer_correct = 0.0
    grounded = 0.0
    answer = trajectory.final_answer or {}
    if format_valid:
        predicted = float(answer["value"])
        expected = float(task.answer["value"])
        tolerance = float(task.answer.get("tolerance", 0.0))
        unit_ok = answer["unit"] == task.answer["unit"]
        answer_correct = float(unit_ok and math.isclose(predicted, expected, abs_tol=tolerance, rel_tol=0.0))
        cited = set(answer["observation_ids"])
        cited_results = [known_observations[item] for item in cited if item in known_observations]
        grounded = float(
            bool(cited)
            and len(cited_results) == len(cited)
            and any(
                result.get("unit") == answer["unit"]
                and isinstance(result.get("scalar"), (int, float))
                and math.isclose(float(result["scalar"]), predicted, abs_tol=tolerance, rel_tol=0.0)
                for result in cited_results
            )
        )

    used_families = {TOOL_FAMILY_BY_NAME.get(call.name) for call in calls if call.result.get("ok")}
    required = set(task.required_tool_families)
    required_family_coverage = 1.0 if not required else len(required & used_families) / len(required)

    expected_steps = max(1, len(task.oracle_steps))
    step_budget = expected_steps + EXPLORATION_ALLOWANCE
    over_budget_steps = max(0, len(calls) - step_budget)
    # Efficiency is diagnostic only. One exploratory call beyond the oracle is
    # explicitly free; further calls lower this value but never lower reward.
    efficiency = max(0.0, 1.0 - over_budget_steps / expected_steps)

    hard_failure: str | None = None
    # Terminal reasons from the harness take priority so format / API failures are
    # not collapsed into execution_failure when no tool call was recorded.
    if trajectory.terminal_reason == "invalid_action":
        hard_failure = "invalid_answer_format"
        format_valid = 0.0
    elif trajectory.terminal_reason == "model_call_error":
        hard_failure = "model_call_error"
    elif not argument_valid:
        hard_failure = "invalid_arguments"
    elif not temporal_valid:
        hard_failure = "temporal_violation"
    elif not execution_valid:
        hard_failure = "execution_failure"
    elif not format_valid:
        hard_failure = "invalid_answer_format"

    if hard_failure:
        total = 0.0
    else:
        # Efficiency is deliberately excluded from scalar reward in m1-v3. It is
        # retained in the vector for cost and reward-hacking diagnostics.
        total = (
            0.50 * answer_correct
            + 0.25 * grounded
            + 0.10 * execution_valid
            + 0.05 * argument_valid
            + 0.05 * temporal_valid
            + 0.05 * required_family_coverage
        )
    return RewardVector(
        execution_valid=execution_valid,
        answer_correct=answer_correct,
        argument_valid=argument_valid,
        temporal_valid=temporal_valid,
        grounded=grounded,
        format_valid=format_valid,
        efficiency=efficiency,
        required_family_coverage=required_family_coverage,
        hard_failure=hard_failure,
        total=round(total, 8),
    )
