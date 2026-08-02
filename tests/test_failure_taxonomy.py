from __future__ import annotations

from pathlib import Path

from fintool_rl.baseline_report import build_baseline_report
from fintool_rl.contracts import TaskSpec, Trajectory
from fintool_rl.database import build_fixture_snapshot
from fintool_rl.failure_taxonomy import classify_failure
from fintool_rl.harness import HarnessRunner, OraclePolicy, TrajectoryStore
from fintool_rl.policies import ActionParseError
from fintool_rl.reward import grade_trajectory
from fintool_rl.tasks import generate_fixture_tasks


def _lookup_task(tasks, family: str) -> TaskSpec:
    return next(task for task in tasks if task.template_family == family)


def test_taxonomy_labels_common_failure_modes(tmp_path: Path):
    db = build_fixture_snapshot(tmp_path / "snap.sqlite")
    tasks = generate_fixture_tasks(db)
    task = _lookup_task(tasks, "financial_fact_lookup")

    guess = Trajectory(
        trajectory_id="g",
        task_id=task.task_id,
        policy_name="guess",
        final_answer={
            "value": task.answer["value"],
            "unit": task.answer["unit"],
            "observation_ids": [],
        },
        terminal_reason="answered",
    )
    guess_reward = grade_trajectory(task, guess)
    assert classify_failure(task, guess, guess_reward).primary == "no_tool_use"

    bad_json = Trajectory(
        trajectory_id="b",
        task_id=task.task_id,
        policy_name="bad",
        actions=[{"kind": "invalid_action", "raw_text": "nope", "error": "bad"}],
        terminal_reason="invalid_action",
    )
    bad_reward = grade_trajectory(task, bad_json)
    assert classify_failure(task, bad_json, bad_reward).primary == "invalid_answer_format"

    # Build a real grounded trajectory then change the unit.
    runner = HarnessRunner(db)
    trajectory, _ = runner.run(task, OraclePolicy())
    trajectory.final_answer = {
        "value": task.answer["value"],
        "unit": "not_the_unit",
        "observation_ids": trajectory.final_answer["observation_ids"],
    }
    unit_reward = grade_trajectory(task, trajectory)
    assert classify_failure(task, trajectory, unit_reward).primary == "wrong_unit"


def test_oracle_baseline_report_is_all_success(tmp_path: Path):
    db = build_fixture_snapshot(tmp_path / "snap.sqlite")
    tasks = generate_fixture_tasks(db)[:10]
    store = TrajectoryStore(tmp_path / "store.sqlite")
    runner = HarnessRunner(db)
    for task in tasks:
        trajectory, reward = runner.run(task, OraclePolicy())
        store.save(trajectory, reward)
    tasks_path = tmp_path / "tasks.jsonl"
    tasks_path.write_text(
        "\n".join(
            __import__("json").dumps(task.to_dict(), sort_keys=True) for task in tasks
        )
        + "\n",
        encoding="utf-8",
    )
    report = build_baseline_report(
        tasks=tasks,
        graded=store.load_graded(),
        db_path=db,
        tasks_path=tasks_path,
        store_path=tmp_path / "store.sqlite",
        policy_name="OraclePolicy",
        max_steps=8,
    )
    assert report["overall"]["n"] == 10
    assert report["overall"]["success_rate"] == 1.0
    assert report["failure_taxonomy"]["primary"] == {"success": 10}


def test_harness_invalid_action_classified_as_format_failure(tmp_path: Path):
    db = build_fixture_snapshot(tmp_path / "snap.sqlite")
    task = generate_fixture_tasks(db)[0]

    class BadPolicy:
        name = "BadPolicy"
        privileged = False

        def reset(self, task):
            return None

        def act(self, observation):
            raise ActionParseError("bad json", raw_text="{")

    trajectory, reward = HarnessRunner(db).run(task, BadPolicy())
    label = classify_failure(task, trajectory, reward)
    assert label.primary == "invalid_answer_format"
