from __future__ import annotations

import pytest

from fintool_rl.contracts import ToolCall, Trajectory
from fintool_rl.gate_b import (
    build_reachability_report,
    calculator_called,
    require_gate_b_settings,
    summarize_task_group,
    trajectory_fingerprint,
)
from fintool_rl.reward import RewardVector


def _reward(*, answer_correct: float, total: float, hard_failure: str | None = None) -> RewardVector:
    return RewardVector(
        execution_valid=1.0,
        answer_correct=answer_correct,
        argument_valid=1.0,
        temporal_valid=1.0,
        grounded=1.0,
        format_valid=1.0,
        efficiency=1.0,
        required_family_coverage=1.0,
        hard_failure=hard_failure,
        total=total,
    )


def _traj(task_id: str, actions: list[dict], *, tool_name: str | None = None) -> Trajectory:
    calls = []
    if tool_name:
        calls.append(
            ToolCall(
                name=tool_name,
                arguments={"symbol": "ALFA"},
                call_id="c1",
                result={"ok": True},
            )
        )
    return Trajectory(
        trajectory_id="t1",
        task_id=task_id,
        policy_name="test",
        actions=actions,
        tool_calls=calls,
        terminal_reason="answered",
    )


def test_identical_samples_have_zero_diversity_and_can_pass_at_k() -> None:
    action = {"kind": "tool", "tool_name": "get_financial_fact", "arguments": {"symbol": "ALFA"}}
    rows = [
        (_traj("task_a", [action], tool_name="get_financial_fact"), _reward(answer_correct=1.0, total=0.9)),
        (_traj("task_a", [action], tool_name="get_financial_fact"), _reward(answer_correct=1.0, total=0.9)),
    ]
    summary = summarize_task_group("task_a", rows)
    assert summary["k"] == 2
    assert summary["pass_at_k"] == 1
    assert summary["unique_trajectories"] == 1
    assert summary["trajectory_diversity"] == 0.5
    assert summary["action_diversity"] == 0.5
    assert summary["reward_variance"] == 0.0
    assert summary["calculator_called_rate"] == 0.0


def test_distinct_actions_raise_diversity_and_calculator_flag() -> None:
    first = _traj(
        "task_b",
        [{"kind": "tool", "tool_name": "get_financial_fact", "arguments": {"metric": "revenue"}}],
        tool_name="get_financial_fact",
    )
    second = _traj(
        "task_b",
        [{"kind": "tool", "tool_name": "calculate_ratio", "arguments": {"scale": 100.0}}],
        tool_name="calculate_ratio",
    )
    rows = [
        (first, _reward(answer_correct=0.0, total=0.3)),
        (second, _reward(answer_correct=1.0, total=0.9)),
    ]
    assert calculator_called(second) is True
    assert calculator_called(first) is False
    assert trajectory_fingerprint(first) != trajectory_fingerprint(second)
    summary = summarize_task_group("task_b", rows)
    assert summary["pass_at_k"] == 1
    assert summary["n_correct"] == 1
    assert summary["unique_trajectories"] == 2
    assert summary["trajectory_diversity"] == 1.0
    assert summary["calculator_called_rate"] == 0.5
    assert summary["reward_variance"] > 0


def test_hard_failure_does_not_count_as_pass_at_k() -> None:
    action = {"kind": "answer", "answer": {"value": 1, "unit": "USD_million", "observation_ids": []}}
    rows = [
        (_traj("task_c", [action]), _reward(answer_correct=1.0, total=0.0, hard_failure="no_tool_use")),
        (_traj("task_c", [action]), _reward(answer_correct=0.0, total=0.2)),
    ]
    summary = summarize_task_group("task_c", rows)
    assert summary["pass_at_k"] == 0
    assert summary["n_correct"] == 0


def test_report_aggregates_task_groups() -> None:
    action = {"kind": "tool", "tool_name": "get_financial_fact", "arguments": {}}
    graded = [
        (_traj("one", [action]), _reward(answer_correct=1.0, total=0.9)),
        (_traj("one", [action]), _reward(answer_correct=0.0, total=0.3)),
        (_traj("two", [action]), _reward(answer_correct=0.0, total=0.1)),
        (_traj("two", [{"kind": "tool", "tool_name": "list_available_periods", "arguments": {}}]), _reward(answer_correct=0.0, total=0.2)),
    ]
    report = build_reachability_report(graded, temperature=0.7, k=2, policy_name="test")
    assert report["n_tasks"] == 2
    assert report["pass_at_k"] == 0.5
    assert report["tasks_with_multiple_trajectories"] == 1


def test_settings_fail_closed() -> None:
    with pytest.raises(ValueError, match="temperature"):
        require_gate_b_settings(temperature=0.0, k=4)
    with pytest.raises(ValueError, match="K>1"):
        require_gate_b_settings(temperature=0.7, k=1)
    require_gate_b_settings(temperature=0.7, k=4)
