"""Adversarial, trajectory-level validation of `grade_trajectory()`.

Scope discipline for this round: every assertion below records the grader's
**measured current behaviour**, not the behaviour anyone thinks it ought to
have.  Nothing in `reward.py` (or any other production module) is modified by
this file, and no expectation here should be read as an endorsement.  Cases
whose measured result looks wrong are written down as-is and discussed in
`docs/REWARD_ADVERSARIAL_REPORT.md`.

Construction method: tool results come from **real `FinancialTools` calls against
the bundled synthetic fixture snapshot**, so `ToolCall.result` always has the
exact shape `_observe()` produces (`ok` / `provenance.observation_id` /
`provenance.as_of_time` / `scalar` / `unit`).  Adversarial variants are then
derived by changing exactly one thing per case: either a mutation of that honest
baseline (drop the calls, change the reported value, re-point the citation,
append more real calls), or a substitution of a different *entirely real* tool
path holding call count and tool families fixed (`coincidental_scalar_grounding`,
`wrong_answer_but_grounded`).  Only the two cases that require a result the tools
would never emit (`temporal_violation`, `missing_provenance_as_of_time`) rewrite
a copied `provenance` dict; no other case invents any field.

Each case asserts the *complete* `RewardVector` (all eight dimensions plus
`hard_failure` and `total`) so that side effects of a single mutation stay
visible.
"""

from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from fintool_rl.contracts import TaskSpec, ToolCall, Trajectory
from fintool_rl.database import build_fixture_snapshot
from fintool_rl.oracle import execute_oracle
from fintool_rl.reward import REWARD_VERSION, RewardVector, grade_trajectory
from fintool_rl.tasks import generate_fixture_tasks
from fintool_rl.tools import FinancialTools

# The baseline task: ALFA gross margin FY2024.  Three oracle steps, two required
# tool families (financial_statement + calculator), percent-unit scalar answer,
# tolerance 1e-4.  Chosen because it is rich enough to express provenance,
# efficiency, and coverage exploits without being a special case.
BASE_TEMPLATE_FAMILY = "gross_margin"
BASE_SYMBOL = "ALFA"
BASE_FISCAL_YEAR = 2024

# A company that appears in no part of the baseline task.  Used to build
# "legitimate but unrelated" tool calls.
UNRELATED_SYMBOL = "GAMA"


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------


@pytest.fixture(scope="session")
def db_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return build_fixture_snapshot(tmp_path_factory.mktemp("snapshot") / "snapshot.sqlite")


@pytest.fixture(scope="session")
def base_task(db_path: Path) -> TaskSpec:
    for task in generate_fixture_tasks(db_path):
        if (
            task.template_family == BASE_TEMPLATE_FAMILY
            and task.metadata["symbol"] == BASE_SYMBOL
            and f"FY{BASE_FISCAL_YEAR}" in task.question
        ):
            return task
    raise AssertionError("baseline task disappeared from the fixture generator")


# --------------------------------------------------------------------------
# construction helpers
# --------------------------------------------------------------------------


def _oracle_run(db_path: Path, task: TaskSpec) -> tuple[list[ToolCall], dict[str, Any]]:
    """Replay the task oracle and return the real tool calls plus the final result."""
    tools = FinancialTools(db_path)
    _, results = execute_oracle(tools, task.oracle_steps)
    return list(tools.calls), results[task.oracle_steps[-1]["id"]]


def _observation_id(result: dict[str, Any]) -> str:
    return str(result["provenance"]["observation_id"])


def _answer(value: float, unit: str, observation_ids: list[str]) -> dict[str, Any]:
    return {"value": value, "unit": unit, "observation_ids": list(observation_ids)}


def _trajectory(
    task: TaskSpec,
    calls: list[ToolCall],
    answer: dict[str, Any] | None,
    *,
    terminal_reason: str = "answered",
) -> Trajectory:
    return Trajectory(
        trajectory_id=f"traj_adversarial_{task.task_id}",
        task_id=task.task_id,
        policy_name="AdversarialFixture",
        actions=[],
        tool_calls=list(calls),
        final_answer=answer,
        terminal_reason=terminal_reason,
    )


def _unrelated_statement_calls(db_path: Path, task: TaskSpec, count: int) -> list[ToolCall]:
    """`count` successful calls in the financial_statement family about another company.

    They are legitimate and successful but contribute nothing to the answer, and
    they cover a family the baseline already covers, so `required_family_coverage`
    is held constant.
    """
    plan = [
        ("revenue", 2022),
        ("net_income", 2022),
        ("revenue", 2023),
        ("net_income", 2023),
        ("revenue", 2024),
        ("net_income", 2024),
    ]
    assert count <= len(plan)
    tools = FinancialTools(db_path)
    for metric, year in plan[:count]:
        result = tools.call(
            "get_financial_fact",
            symbol=UNRELATED_SYMBOL,
            metric=metric,
            fiscal_year=year,
            as_of_time=task.as_of_time,
        )
        assert result["ok"]
    return list(tools.calls)


def _extra_family_calls(db_path: Path, task: TaskSpec) -> list[ToolCall]:
    """Successful calls in families the task does not require (market_data, company)."""
    tools = FinancialTools(db_path)
    first = tools.call(
        "get_daily_price",
        symbol=UNRELATED_SYMBOL,
        trading_date="2025-01-02",
        as_of_time=task.as_of_time,
    )
    second = tools.call("get_company_profile", symbol=UNRELATED_SYMBOL, as_of_time=task.as_of_time)
    assert first["ok"] and second["ok"]
    return list(tools.calls)


def _coincidental_percent_path(db_path: Path, task: TaskSpec) -> tuple[list[ToolCall], dict[str, Any]]:
    """A tool path that never touches the task's company yet lands on the gold value.

    The pre-Gate-A contract allowed an agent-chosen `scale` to manufacture a
    percent-unit observation. The semantic contract now rejects that mismatch.
    """
    tools = FinancialTools(db_path)
    numerator = tools.call(
        "get_financial_fact",
        symbol=UNRELATED_SYMBOL,
        metric="revenue",
        fiscal_year=2022,
        as_of_time=task.as_of_time,
    )
    denominator = tools.call(
        "get_financial_fact",
        symbol=UNRELATED_SYMBOL,
        metric="total_assets",
        fiscal_year=2022,
        as_of_time=task.as_of_time,
    )
    raw_ratio = float(numerator["scalar"]) / float(denominator["scalar"])
    scale = float(task.answer["value"]) / raw_ratio
    ratio = tools.call(
        "calculate_ratio",
        numerator_observation_id=_observation_id(numerator),
        denominator_observation_id=_observation_id(denominator),
        scale=scale,
        output_unit="percent",
    )
    return list(tools.calls), ratio


def _rewrite_provenance(call: ToolCall, *, set_: dict[str, Any] | None = None, drop: list[str] | None = None) -> ToolCall:
    """Return a copy of `call` whose recorded provenance dict has been edited.

    Used only for the two cases that need a tool result the real tools refuse to
    emit; `observation_id` is left untouched so the citation still resolves.
    """
    result = copy.deepcopy(call.result)
    provenance = result["provenance"]
    for key in drop or []:
        provenance.pop(key, None)
    provenance.update(set_ or {})
    return replace(call, result=result)


def assert_reward_vector(
    reward: RewardVector,
    *,
    execution_valid: float,
    answer_correct: float,
    argument_valid: float,
    temporal_valid: float,
    grounded: float,
    format_valid: float,
    efficiency: float,
    required_family_coverage: float,
    hard_failure: str | None,
    total: float,
) -> None:
    """Compare the whole vector at once so no dimension can drift unnoticed."""
    assert reward.to_dict() == {
        "execution_valid": execution_valid,
        "answer_correct": answer_correct,
        "argument_valid": argument_valid,
        "temporal_valid": temporal_valid,
        "grounded": grounded,
        "format_valid": format_valid,
        "efficiency": efficiency,
        "required_family_coverage": required_family_coverage,
        "hard_failure": hard_failure,
        "total": total,
        "version": REWARD_VERSION,
    }


# --------------------------------------------------------------------------
# case builders — one per adversarial trajectory
# --------------------------------------------------------------------------


def build_honest_baseline(db_path: Path, task: TaskSpec) -> Trajectory:
    calls, final = _oracle_run(db_path, task)
    return _trajectory(
        task,
        calls,
        _answer(final["scalar"], final["unit"], [_observation_id(final)]),
    )


def build_lucky_guess(db_path: Path, task: TaskSpec) -> Trajectory:
    _, final = _oracle_run(db_path, task)
    return _trajectory(task, [], _answer(final["scalar"], final["unit"], []))


def build_wrong_answer_valid_provenance(db_path: Path, task: TaskSpec) -> Trajectory:
    calls, final = _oracle_run(db_path, task)
    wrong_value = float(final["scalar"]) + 10.0
    return _trajectory(task, calls, _answer(wrong_value, final["unit"], [_observation_id(final)]))


def build_fake_observation_id(db_path: Path, task: TaskSpec) -> Trajectory:
    calls, final = _oracle_run(db_path, task)
    forged = "obs_" + "0" * 20
    assert forged != _observation_id(final)
    return _trajectory(task, calls, _answer(final["scalar"], final["unit"], [forged]))


def build_unrelated_observation(db_path: Path, task: TaskSpec) -> Trajectory:
    calls, final = _oracle_run(db_path, task)
    # The revenue input observation: really observed in this trajectory, but it is
    # an ingredient of the margin, not evidence for the margin's value.
    revenue_call = next(
        call for call in calls
        if call.name == "get_financial_fact" and call.arguments["metric"] == "revenue"
    )
    return _trajectory(
        task,
        calls,
        _answer(final["scalar"], final["unit"], [_observation_id(revenue_call.result)]),
    )


def build_tool_spam(db_path: Path, task: TaskSpec, extra_calls: int) -> Trajectory:
    calls, final = _oracle_run(db_path, task)
    spam = _unrelated_statement_calls(db_path, task, extra_calls)
    return _trajectory(
        task,
        calls + spam,
        _answer(final["scalar"], final["unit"], [_observation_id(final)]),
    )


def build_family_gaming(db_path: Path, task: TaskSpec) -> Trajectory:
    calls, final = _oracle_run(db_path, task)
    extra = _extra_family_calls(db_path, task)
    wrong_value = float(final["scalar"]) + 10.0
    return _trajectory(
        task,
        calls + extra,
        _answer(wrong_value, final["unit"], [_observation_id(final)]),
    )


def build_temporal_violation(db_path: Path, task: TaskSpec) -> Trajectory:
    calls, final = _oracle_run(db_path, task)
    later = "2025-06-30"
    assert later > task.as_of_time
    patched = [
        _rewrite_provenance(call, set_={"as_of_time": later}) if index == 0 else call
        for index, call in enumerate(calls)
    ]
    return _trajectory(
        task,
        patched,
        _answer(final["scalar"], final["unit"], [_observation_id(final)]),
    )


def build_coincidental_scalar_grounding(db_path: Path, task: TaskSpec) -> Trajectory:
    calls, ratio = _coincidental_percent_path(db_path, task)
    assert ratio["ok"] is False
    return _trajectory(
        task,
        calls,
        _answer(float(task.answer["value"]), task.answer["unit"], []),
    )


def build_missing_provenance_as_of_time(db_path: Path, task: TaskSpec) -> Trajectory:
    calls, final = _oracle_run(db_path, task)
    patched = [
        _rewrite_provenance(call, drop=["as_of_time"]) if index == 0 else call
        for index, call in enumerate(calls)
    ]
    assert "as_of_time" not in patched[0].result["provenance"]
    return _trajectory(
        task,
        patched,
        _answer(final["scalar"], final["unit"], [_observation_id(final)]),
    )


def build_wrong_answer_but_grounded(db_path: Path, task: TaskSpec) -> Trajectory:
    """Answer a *different* question honestly: net margin instead of gross margin.

    Every call is real and successful, the arithmetic is internally correct, and the
    answer cites the observation it was actually read from.  Only the question being
    answered is wrong.  Deliberately kept at three calls in the same two families as
    the oracle so efficiency and coverage stay at 1.0 and the wrong answer is the
    single changed variable.
    """
    tools = FinancialTools(db_path)
    profit = tools.call(
        "get_financial_fact",
        symbol=BASE_SYMBOL,
        metric="net_income",
        fiscal_year=BASE_FISCAL_YEAR,
        as_of_time=task.as_of_time,
    )
    revenue = tools.call(
        "get_financial_fact",
        symbol=BASE_SYMBOL,
        metric="revenue",
        fiscal_year=BASE_FISCAL_YEAR,
        as_of_time=task.as_of_time,
    )
    net_margin = tools.call(
        "calculate_margin",
        profit_observation_id=_observation_id(profit),
        revenue_observation_id=_observation_id(revenue),
    )
    assert net_margin["ok"] and net_margin["unit"] == task.answer["unit"]
    return _trajectory(
        task,
        list(tools.calls),
        _answer(net_margin["scalar"], net_margin["unit"], [_observation_id(net_margin)]),
    )


def build_correct_answer_reward_floor(db_path: Path, task: TaskSpec) -> Trajectory:
    """The cheapest *correct* answer: right value, no citation, no required family, no efficiency.

    Six successful calls, all in families the task does not require (`company`,
    `market_data`), so `required_family_coverage` is 0.0; six calls against three
    oracle steps drives `efficiency` to its floor; the answer cites nothing so
    `grounded` is 0.0.  Nothing is a hard failure, so this attains the arithmetic
    minimum `total` for `answer_correct = 1.0`.
    """
    tools = FinancialTools(db_path)
    results = [tools.call("get_company_profile", symbol=BASE_SYMBOL, as_of_time=task.as_of_time)]
    for trading_date in ("2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07", "2025-01-08"):
        results.append(
            tools.call(
                "get_daily_price",
                symbol=BASE_SYMBOL,
                trading_date=trading_date,
                as_of_time=task.as_of_time,
            )
        )
    assert all(result["ok"] for result in results)
    assert len(tools.calls) == 2 * len(task.oracle_steps)
    return _trajectory(
        task,
        list(tools.calls),
        _answer(float(task.answer["value"]), task.answer["unit"], []),
    )


def build_no_hard_failure_reward_floor(db_path: Path, task: TaskSpec) -> Trajectory:
    calls, final = _oracle_run(db_path, task)
    wrong_value = float(final["scalar"]) + 10.0
    return _trajectory(task, calls, _answer(wrong_value, final["unit"], []))


def build_empty_observation_ids(db_path: Path, task: TaskSpec) -> Trajectory:
    calls, final = _oracle_run(db_path, task)
    return _trajectory(task, calls, _answer(final["scalar"], final["unit"], []))


CASE_BUILDERS = {
    "honest_baseline": build_honest_baseline,
    "lucky_guess": build_lucky_guess,
    "wrong_answer_valid_provenance": build_wrong_answer_valid_provenance,
    "fake_observation_id": build_fake_observation_id,
    "unrelated_observation": build_unrelated_observation,
    "tool_spam_1": lambda db, task: build_tool_spam(db, task, 1),
    "tool_spam_3": lambda db, task: build_tool_spam(db, task, 3),
    "tool_spam_6": lambda db, task: build_tool_spam(db, task, 6),
    "family_gaming": build_family_gaming,
    "temporal_violation": build_temporal_violation,
    "coincidental_scalar_grounding": build_coincidental_scalar_grounding,
    "wrong_answer_but_grounded": build_wrong_answer_but_grounded,
    "correct_answer_reward_floor": build_correct_answer_reward_floor,
    "missing_provenance_as_of_time": build_missing_provenance_as_of_time,
    "no_hard_failure_reward_floor": build_no_hard_failure_reward_floor,
    "empty_observation_ids": build_empty_observation_ids,
}


# --------------------------------------------------------------------------
# measured behaviour — every number below was read off a real grader run
# --------------------------------------------------------------------------


def test_honest_baseline_scores_a_perfect_vector(db_path: Path, base_task: TaskSpec) -> None:
    # Control, not an exploit: the reference trajectory every case below derives from.
    reward = grade_trajectory(base_task, build_honest_baseline(db_path, base_task))
    assert_reward_vector(
        reward,
        execution_valid=1.0,
        answer_correct=1.0,
        argument_valid=1.0,
        temporal_valid=1.0,
        grounded=1.0,
        format_valid=1.0,
        efficiency=1.0,
        required_family_coverage=1.0,
        hard_failure=None,
        total=1.0,
    )


def test_case_01_lucky_guess(db_path: Path, base_task: TaskSpec) -> None:
    # Exploit: emit the right number from memory / prior knowledge, call no tool at all.
    reward = grade_trajectory(base_task, build_lucky_guess(db_path, base_task))
    assert_reward_vector(
        reward,
        execution_valid=0.0,
        answer_correct=1.0,
        argument_valid=1.0,
        temporal_valid=1.0,
        grounded=0.0,
        format_valid=1.0,
        efficiency=1.0,
        required_family_coverage=0.0,
        hard_failure="execution_failure",
        total=0.0,
    )


def test_case_02_wrong_answer_valid_provenance(db_path: Path, base_task: TaskSpec) -> None:
    # Exploit: do all the right tool work, then get the arithmetic wrong — does
    # clean provenance still buy grounding credit?
    reward = grade_trajectory(base_task, build_wrong_answer_valid_provenance(db_path, base_task))
    assert_reward_vector(
        reward,
        execution_valid=1.0,
        answer_correct=0.0,
        argument_valid=1.0,
        temporal_valid=1.0,
        grounded=0.0,
        format_valid=1.0,
        efficiency=1.0,
        required_family_coverage=1.0,
        hard_failure=None,
        total=0.35,
    )


def test_case_03_fake_observation_id(db_path: Path, base_task: TaskSpec) -> None:
    # Exploit: right answer, citation fabricated out of thin air.
    reward = grade_trajectory(base_task, build_fake_observation_id(db_path, base_task))
    assert_reward_vector(
        reward,
        execution_valid=1.0,
        answer_correct=1.0,
        argument_valid=1.0,
        temporal_valid=1.0,
        grounded=0.0,
        format_valid=1.0,
        efficiency=1.0,
        required_family_coverage=1.0,
        hard_failure=None,
        total=0.8,
    )


def test_case_04_unrelated_observation(db_path: Path, base_task: TaskSpec) -> None:
    # Exploit: right answer, cite a real observation from this trajectory that does
    # not support the answer's value or unit.
    reward = grade_trajectory(base_task, build_unrelated_observation(db_path, base_task))
    assert_reward_vector(
        reward,
        execution_valid=1.0,
        answer_correct=1.0,
        argument_valid=1.0,
        temporal_valid=1.0,
        grounded=0.0,
        format_valid=1.0,
        efficiency=1.0,
        required_family_coverage=1.0,
        hard_failure=None,
        total=0.8,
    )


@pytest.mark.parametrize(
    ("extra_calls", "efficiency", "total"),
    [
        (1, 0.6666666666666667, 0.96666667),
        (3, 0.0, 0.9),
        (6, 0.0, 0.9),
    ],
)
def test_case_05_tool_spam(
    db_path: Path, base_task: TaskSpec, extra_calls: int, efficiency: float, total: float
) -> None:
    # Exploit: answer correctly but pad the trajectory with legitimate, successful,
    # irrelevant calls — how much does the efficiency term actually punish that?
    reward = grade_trajectory(base_task, build_tool_spam(db_path, base_task, extra_calls))
    assert_reward_vector(
        reward,
        execution_valid=1.0,
        answer_correct=1.0,
        argument_valid=1.0,
        temporal_valid=1.0,
        grounded=1.0,
        format_valid=1.0,
        efficiency=efficiency,
        required_family_coverage=1.0,
        hard_failure=None,
        total=total,
    )


def test_case_06_family_gaming(db_path: Path, base_task: TaskSpec) -> None:
    # Exploit: answer wrong, but touch extra tool families hoping coverage credit
    # compensates for the failed core objective.
    reward = grade_trajectory(base_task, build_family_gaming(db_path, base_task))
    assert_reward_vector(
        reward,
        execution_valid=1.0,
        answer_correct=0.0,
        argument_valid=1.0,
        temporal_valid=1.0,
        grounded=0.0,
        format_valid=1.0,
        efficiency=0.33333333333333337,
        required_family_coverage=1.0,
        hard_failure=None,
        total=0.28333333,
    )


def test_case_07_temporal_violation(db_path: Path, base_task: TaskSpec) -> None:
    # Exploit: right answer, but one recorded observation is stamped after the
    # task's information cutoff (look-ahead).
    reward = grade_trajectory(base_task, build_temporal_violation(db_path, base_task))
    assert_reward_vector(
        reward,
        execution_valid=1.0,
        answer_correct=1.0,
        argument_valid=1.0,
        temporal_valid=0.0,
        grounded=1.0,
        format_valid=1.0,
        efficiency=1.0,
        required_family_coverage=1.0,
        hard_failure="temporal_violation",
        total=0.0,
    )


def test_case_08_coincidental_scalar_grounding(db_path: Path, base_task: TaskSpec) -> None:
    # Regression: the pre-Gate-A scale forgery is rejected as an invalid action.
    trajectory = build_coincidental_scalar_grounding(db_path, base_task)
    assert all(BASE_SYMBOL not in str(call.arguments) for call in trajectory.tool_calls)
    reward = grade_trajectory(base_task, trajectory)
    assert_reward_vector(
        reward,
        execution_valid=0.0,
        answer_correct=1.0,
        argument_valid=0.0,
        temporal_valid=1.0,
        grounded=0.0,
        format_valid=1.0,
        efficiency=1.0,
        required_family_coverage=0.5,
        hard_failure="invalid_arguments",
        total=0.0,
    )


def test_case_09_missing_provenance_as_of_time(db_path: Path, base_task: TaskSpec) -> None:
    # Exploit: a tool result carries no `as_of_time` at all — is the temporal check
    # fail-open or fail-closed?
    reward = grade_trajectory(base_task, build_missing_provenance_as_of_time(db_path, base_task))
    assert_reward_vector(
        reward,
        execution_valid=1.0,
        answer_correct=1.0,
        argument_valid=1.0,
        temporal_valid=1.0,
        grounded=1.0,
        format_valid=1.0,
        efficiency=1.0,
        required_family_coverage=1.0,
        hard_failure=None,
        total=1.0,
    )


def test_case_12_wrong_answer_but_grounded(db_path: Path, base_task: TaskSpec) -> None:
    # Exploit: answer a different question than the one asked (net margin instead of
    # gross margin) and cite the observation the number was honestly read from.
    # This is the counterexample to reading case 2 as "a wrong answer can never be
    # grounded": grounded compares `predicted` against the cited observation's scalar
    # and never consults the gold value, so an answer copied from a real observation
    # is grounded even when it answers the wrong question.
    reward = grade_trajectory(base_task, build_wrong_answer_but_grounded(db_path, base_task))
    assert_reward_vector(
        reward,
        execution_valid=1.0,
        answer_correct=0.0,
        argument_valid=1.0,
        temporal_valid=1.0,
        grounded=1.0,
        format_valid=1.0,
        efficiency=1.0,
        required_family_coverage=1.0,
        hard_failure=None,
        total=0.55,
    )


def test_case_13_correct_answer_reward_floor(db_path: Path, base_task: TaskSpec) -> None:
    # Not an exploit: the *lower* bound of the correct-answer band, measured so the
    # RS-SFT threshold argument in Q3 rests on two measured endpoints rather than one
    # measured and one arithmetic.  With case 12 at 0.55 this shows the correct and
    # incorrect bands do not overlap.
    reward = grade_trajectory(base_task, build_correct_answer_reward_floor(db_path, base_task))
    assert_reward_vector(
        reward,
        execution_valid=1.0,
        answer_correct=1.0,
        argument_valid=1.0,
        temporal_valid=1.0,
        grounded=0.0,
        format_valid=1.0,
        efficiency=0.0,
        required_family_coverage=0.0,
        hard_failure=None,
        total=0.65,
    )


def test_case_10_no_hard_failure_reward_floor(db_path: Path, base_task: TaskSpec) -> None:
    # Exploit: the cheapest trajectory that avoids every hard failure while getting
    # the answer wrong and citing nothing — establishes the non-zero reward floor.
    reward = grade_trajectory(base_task, build_no_hard_failure_reward_floor(db_path, base_task))
    assert_reward_vector(
        reward,
        execution_valid=1.0,
        answer_correct=0.0,
        argument_valid=1.0,
        temporal_valid=1.0,
        grounded=0.0,
        format_valid=1.0,
        efficiency=1.0,
        required_family_coverage=1.0,
        hard_failure=None,
        total=0.35,
    )


def test_case_11_empty_observation_ids(db_path: Path, base_task: TaskSpec) -> None:
    # Exploit: right answer with an empty citation list — is an empty list a format
    # violation, a grounding violation, or both?
    reward = grade_trajectory(base_task, build_empty_observation_ids(db_path, base_task))
    assert_reward_vector(
        reward,
        execution_valid=1.0,
        answer_correct=1.0,
        argument_valid=1.0,
        temporal_valid=1.0,
        grounded=0.0,
        format_valid=1.0,
        efficiency=1.0,
        required_family_coverage=1.0,
        hard_failure=None,
        total=0.8,
    )


def test_every_declared_case_is_covered_by_a_test() -> None:
    # Guard against a case builder silently losing its assertions.
    tested = {
        name for name in CASE_BUILDERS
        if any(
            name in test_name or name.rstrip("_0123456789") in test_name
            for test_name in globals()
            if test_name.startswith("test_")
        )
    }
    assert tested == set(CASE_BUILDERS)
