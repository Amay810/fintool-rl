from __future__ import annotations

from dataclasses import replace

import pytest

from fintool_rl.contracts import TaskSpec, Trajectory
from fintool_rl.readiness import (
    ReadinessThresholds,
    analyze_readiness,
    expected_drgrpo_mass,
    mixed_probability,
    pass_at_k,
    terminal_success,
)
from fintool_rl.reward import RewardVector


def _reward(success: bool) -> RewardVector:
    value = float(success)
    return RewardVector(
        execution_valid=value,
        answer_correct=value,
        argument_valid=1.0,
        temporal_valid=1.0,
        grounded=value,
        format_valid=1.0,
        efficiency=1.0,
        required_family_coverage=value,
        hard_failure=None if success else "execution_failure",
        total=value,
    )


def _task(index: int, *, depth: int = 6) -> TaskSpec:
    return TaskSpec(
        task_id=f"task_{index}",
        question="q",
        split="train",
        as_of_time="2025-01-01",
        difficulty="compositional",
        template_family=f"family_{index % 2}",
        answer={"value": 1.0, "unit": "percent", "tolerance": 0.0},
        oracle_steps=[{"id": str(i)} for i in range(depth)],
        metadata={"graph_depth": depth},
    )


def test_grpo_statistics_have_distinct_meanings():
    assert pass_at_k(32, 0, 8) == 0.0
    assert pass_at_k(32, 32, 8) == 1.0
    assert mixed_probability(0.5, 8) == pytest.approx(254 / 256)
    assert expected_drgrpo_mass(0.5, 8) == pytest.approx(3.5)


def test_terminal_success_ignores_shaped_total():
    reward = replace(_reward(True), total=0.6)
    assert terminal_success(reward) == 1
    assert terminal_success(_reward(False)) == 0


def test_readiness_requires_tokens_and_uses_binary_outcome():
    tasks = [_task(0), _task(1)]
    graded = []
    for task in tasks:
        for rollout in range(4):
            success = rollout % 2 == 0
            graded.append((Trajectory(
                trajectory_id=f"{task.task_id}_{rollout}",
                task_id=task.task_id,
                policy_name="policy",
                generated_tokens=20,
                policy_version="sft-v1",
            ), _reward(success)))
    report = analyze_readiness(tasks, graded, thresholds=ReadinessThresholds(
        min_samples_per_task=4,
        min_band_tasks_per_stratum=1,
        min_band_tasks_total=2,
        opportunity_waste=0.0,
        icc_lower_bound=-1.0,
        bootstrap_samples=20,
    ))
    assert report["gates"]["token_accounting_complete"] is True
    assert report["gates"]["learnability"] is True
    assert report["overall_go"] is True
    assert report["tasks"][0]["expected_drgrpo_mass"] == pytest.approx(3.5)


def test_readiness_refuses_non_train_tasks():
    task = _task(0)
    task.split = "dev"
    with pytest.raises(ValueError, match="train-only"):
        analyze_readiness([task], [])
