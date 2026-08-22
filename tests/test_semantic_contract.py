from __future__ import annotations

from pathlib import Path

from fintool_rl.contracts import (
    EXACT_SCALAR_TOLERANCE,
    PERCENTAGE_TOLERANCE,
    Trajectory,
)
from fintool_rl.database import build_fixture_snapshot
from fintool_rl.oracle import execute_oracle
from fintool_rl.reward import grade_trajectory
from fintool_rl.tasks import generate_fixture_tasks, generate_snapshot_tasks
from fintool_rl.tools import FinancialTools


def test_generated_derived_tasks_state_semantic_contract_without_formatting(tmp_path: Path):
    db = build_fixture_snapshot(tmp_path / "snapshot.sqlite")
    snapshot_tasks = generate_snapshot_tasks(db, {"ALFA": "test"})
    fixture_tasks = generate_fixture_tasks(db)
    snapshot_percent_tasks = [task for task in snapshot_tasks if task.answer["unit"] == "percent"]
    fixture_derived_tasks = [
        task for task in fixture_tasks
        if task.template_family in {
            "gross_margin",
            "year_over_year_growth",
            "liabilities_to_assets",
            "price_return",
        }
    ]
    percent_tasks = snapshot_percent_tasks + fixture_derived_tasks
    assert snapshot_percent_tasks
    assert {task.template_family for task in fixture_derived_tasks} == {
        "gross_margin",
        "year_over_year_growth",
        "liabilities_to_assets",
        "price_return",
    }
    assert all(task.metadata["generator"] == "sec_snapshot_v2" for task in snapshot_tasks)
    assert all(task.answer["unit"] == "percent" for task in percent_tasks)
    assert all(task.answer["tolerance"] == PERCENTAGE_TOLERANCE for task in percent_tasks)
    assert all(
        not any(marker in task.question.lower() for marker in ("decimal", "round to", "format to"))
        for task in percent_tasks
    )
    exact_tasks = [task for task in snapshot_tasks if task.answer["unit"] == "USD_million"]
    assert exact_tasks
    assert all(task.answer["tolerance"] == EXACT_SCALAR_TOLERANCE for task in exact_tasks)


def test_metric_discoverability_and_invalid_metric_are_consistent(tmp_path: Path):
    db = build_fixture_snapshot(tmp_path / "snapshot.sqlite")
    tools = FinancialTools(db)
    periods = tools.call(
        "list_available_periods", symbol="ALFA", metric="gross_profit", as_of_time="2025-03-31"
    )
    assert periods["ok"]
    assert periods["data"]["metric"] == "gross_profit"
    invalid = tools.call(
        "get_financial_fact",
        symbol="ALFA",
        metric="gross_margin",
        fiscal_year=2024,
        as_of_time="2025-03-31",
    )
    assert invalid == {
        "ok": False,
        "error": "invalid_arguments",
        "detail": "get_financial_fact.metric: must be one of ['revenue', 'gross_profit', 'net_income', 'total_assets', 'total_liabilities']",
    }


def test_percentage_tolerance_is_absolute_in_canonical_unit(tmp_path: Path):
    db = build_fixture_snapshot(tmp_path / "snapshot.sqlite")
    task = next(task for task in generate_snapshot_tasks(db, {"ALFA": "test"}) if task.template_family == "gross_margin")
    tools = FinancialTools(db)
    _, results = execute_oracle(tools, task.oracle_steps)
    final = results[task.oracle_steps[-1]["id"]]
    observation_id = final["provenance"]["observation_id"]

    def grade(value: float, unit: str = "percent"):
        return grade_trajectory(
            task,
            Trajectory(
                trajectory_id="tolerance-test",
                task_id=task.task_id,
                policy_name="test",
                tool_calls=list(tools.calls),
                final_answer={"value": value, "unit": unit, "observation_ids": [observation_id]},
                terminal_reason="answered",
            ),
        )

    expected = float(task.answer["value"])
    tolerance = float(task.answer["tolerance"])
    assert grade(float(final["scalar"])).answer_correct == 1.0
    assert grade(expected + tolerance * 0.99).answer_correct == 1.0
    assert grade(expected + tolerance * 1.01).answer_correct == 0.0
    assert grade(expected / 100.0).answer_correct == 0.0
    assert grade(expected / 100.0, unit="ratio").answer_correct == 0.0
