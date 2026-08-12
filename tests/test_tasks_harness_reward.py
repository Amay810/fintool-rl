from __future__ import annotations

from pathlib import Path

import pytest

from fintool_rl.contracts import AgentAction, TaskSpec, Trajectory
from fintool_rl.database import build_fixture_snapshot
from fintool_rl.failure_taxonomy import classify_failure
from fintool_rl.harness import AgentObservation, HarnessRunner, OraclePolicy, ReplayPolicy
from fintool_rl.reward import REWARD_VERSION, grade_trajectory
from fintool_rl.tasks import (
    assert_no_fact_leakage,
    generate_fixture_tasks,
    generate_growth_of_growth_tasks,
    generate_long_graph_tasks,
    select_split_targets,
)
from fintool_rl.tools import FinancialTools


@pytest.fixture()
def environment(tmp_path: Path):
    db = build_fixture_snapshot(tmp_path / "snapshot.sqlite")
    tasks = generate_fixture_tasks(db)
    return db, tasks


def test_generated_tasks_are_company_disjoint_and_oracle_backed(environment):
    _, tasks = environment
    assert len(tasks) == 85
    assert_no_fact_leakage(tasks)
    assert {task.split for task in tasks} == {"train", "dev", "test", "challenge"}
    assert all(task.oracle_steps and "tolerance" in task.answer for task in tasks)


def test_oracle_policy_gets_full_reward(environment):
    db, tasks = environment
    runner = HarnessRunner(db)
    for task in tasks:
        trajectory, reward = runner.run(task, OraclePolicy())
        assert trajectory.terminal_reason == "answered"
        assert reward.hard_failure is None
        assert reward.total == 1.0
        if task.template_family == "liabilities_to_assets":
            assert task.answer["unit"] == "percent"


def test_efficiency_is_budgeted_diagnostic_not_success_or_reward_gate(environment):
    db, tasks = environment
    task = next(task for task in tasks if task.template_family == "financial_fact_lookup")
    runner = HarnessRunner(db)
    trajectory, _ = runner.run(task, OraclePolicy())
    # Duplicate successful calls to exceed oracle steps + one exploration call.
    trajectory.tool_calls.extend([trajectory.tool_calls[0], trajectory.tool_calls[0]])
    reward = grade_trajectory(task, trajectory)
    label = classify_failure(task, trajectory, reward)
    assert reward.version == REWARD_VERSION == "m1-v3"
    assert reward.efficiency == 0.0
    assert reward.total == 1.0
    assert label.primary == "correct_but_inefficient"


def test_year_over_year_pairs_are_adjacent(environment):
    _, tasks = environment
    for task in tasks:
        if task.template_family != "year_over_year_growth":
            continue
        years = sorted(int(key.split(":")[-1]) for key in task.metadata["fact_keys"])
        assert years[1] == years[0] + 1


def test_non_privileged_policy_never_observes_hidden_gold(environment):
    db, tasks = environment

    class SpyPolicy:
        name = "SpyPolicy"
        privileged = False

        def __init__(self):
            self.reset_payload = None
            self.observation = None

        def reset(self, task):
            self.reset_payload = task

        def act(self, observation: AgentObservation):
            self.observation = observation
            return AgentAction.final(0.0, "unknown", [])

    task = tasks[0]
    policy = SpyPolicy()
    HarnessRunner(db).run(task, policy)
    rendered = str(policy.reset_payload) + str(policy.observation)
    assert "oracle_steps" not in rendered
    assert "tolerance" not in rendered
    assert "fact_keys" not in rendered


def test_correct_guess_without_tools_receives_zero_reward(environment):
    _, tasks = environment
    task = tasks[0]
    trajectory = Trajectory(
        trajectory_id="guess",
        task_id=task.task_id,
        policy_name="GuessPolicy",
        final_answer={
            "value": task.answer["value"],
            "unit": task.answer["unit"],
            "observation_ids": [],
        },
        terminal_reason="answered",
    )
    reward = grade_trajectory(task, trajectory)
    assert reward.answer_correct == 1.0
    assert reward.execution_valid == 0.0
    assert reward.grounded == 0.0
    assert reward.hard_failure == "execution_failure"
    assert reward.total == 0.0


def test_unrelated_existing_observation_does_not_ground_a_correct_guess(environment):
    db, tasks = environment
    task = next(task for task in tasks if task.template_family == "financial_fact_lookup")
    tools = FinancialTools(db)
    unrelated = tools.call("get_company_profile", symbol="ALFA", as_of_time=task.as_of_time)
    trajectory = Trajectory(
        trajectory_id="unrelated-citation",
        task_id=task.task_id,
        policy_name="RewardHacker",
        tool_calls=tools.calls,
        final_answer={
            "value": task.answer["value"],
            "unit": task.answer["unit"],
            "observation_ids": [unrelated["provenance"]["observation_id"]],
        },
        terminal_reason="answered",
    )
    reward = grade_trajectory(task, trajectory)
    assert reward.answer_correct == 1.0
    assert reward.grounded == 0.0


def test_oracle_action_sequence_is_replayable(environment):
    db, tasks = environment
    task = next(task for task in tasks if task.difficulty == "multi_tool")
    runner = HarnessRunner(db)
    original, original_reward = runner.run(task, OraclePolicy())
    replayed, replay_reward = runner.run(task, ReplayPolicy(original.actions))
    assert original_reward.to_dict() == replay_reward.to_dict()
    assert [call.result for call in original.tool_calls] == [call.result for call in replayed.tool_calls]


def test_fact_leakage_audit_fails_closed(environment):
    _, tasks = environment
    original = tasks[0]
    clone = TaskSpec.from_dict(original.to_dict())
    clone.task_id = "leaked"
    clone.split = "test"
    with pytest.raises(ValueError, match="fact leakage"):
        assert_no_fact_leakage([original, clone])


def test_split_target_selection_is_exact_deterministic_and_family_balanced(environment):
    _, tasks = environment
    targets = {"train": 10, "dev": 5, "test": 5}
    first = select_split_targets(tasks, targets)
    second = select_split_targets(reversed(tasks), targets)
    assert [task.task_id for task in first] == [task.task_id for task in second]
    assert {split: sum(task.split == split for task in first) for split in targets} == targets
    assert len({task.template_family for task in first if task.split == "train"}) >= 3


def test_growth_of_growth_probe_has_six_replayable_calls(environment):
    db, _ = environment
    tasks = generate_growth_of_growth_tasks(db, {"GAMA": "dev"}, split="dev", limit=2)
    assert len(tasks) == 2
    assert all(task.template_family == "growth_of_growth" for task in tasks)
    assert all(len(task.oracle_steps) == 6 for task in tasks)
    runner = HarnessRunner(db, max_steps=8)
    for task in tasks:
        _, reward = runner.run(task, OraclePolicy())
        assert reward.total == 1.0


def test_long_graph_pool_has_discovery_reuse_and_seven_call_families(environment):
    db, _ = environment
    tasks = generate_long_graph_tasks(
        db,
        {"ALFA": "train", "BETA": "train", "GAMA": "dev", "DELT": "test"},
    )
    assert {"latest_period_growth_discovery", "growth_of_growth", "gross_margin_change"} <= {
        task.template_family for task in tasks
    }
    assert any(task.metadata["discovery_required"] for task in tasks)
    assert any(task.metadata["observation_reuse_count"] for task in tasks)
    assert any(len(task.oracle_steps) == 7 for task in tasks)
    runner = HarnessRunner(db, max_steps=8)
    for task in tasks:
        _, reward = runner.run(task, OraclePolicy())
        assert reward.total == 1.0
