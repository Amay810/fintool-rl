from __future__ import annotations

import json

import pytest

from fintool_rl.contracts import Trajectory
from fintool_rl.harness import AgentObservation, HarnessRunner
from fintool_rl.policies import (
    SYSTEM_PROMPT,
    ActionParseError,
    JsonActionPolicy,
    ModelCallError,
    parse_action,
)
from fintool_rl.reward import grade_trajectory


def test_parse_tool_and_answer_actions():
    tool = parse_action('{"kind":"tool","tool_name":"get_company_profile","arguments":{"symbol":"ALFA","as_of_time":"2025-03-31"}}')
    assert tool.kind == "tool"
    assert tool.tool_name == "get_company_profile"
    answer = parse_action(
        '```json\n{"kind":"answer","answer":{"value":1.0,"unit":"percent","observation_ids":["obs_x"]}}\n```'
    )
    assert answer.answer == {"value": 1.0, "unit": "percent", "observation_ids": ["obs_x"]}


def test_parse_rejects_non_json_or_ambiguous_payloads():
    with pytest.raises(ActionParseError) as exc_info:
        parse_action("I think the answer is 4")
    assert "I think the answer is 4" in exc_info.value.raw_text
    with pytest.raises(ActionParseError):
        parse_action('{"kind":"tool","tool_name":4,"arguments":{}}')


def test_system_prompt_states_protocol_v1_action_contract():
    prompt = SYSTEM_PROMPT.lower()

    assert "exactly one action" in prompt
    assert "wait for the environment" in prompt
    assert "observation" in prompt
    assert "never emit multiple json objects" in prompt
    assert "jsonl" in prompt
    assert 'the tool-action key is exactly "arguments"' in prompt
    assert 'do not use "tool_arguments"' in prompt
    assert "no markdown" in prompt


def test_json_policy_prompt_contains_only_public_task_and_observations():
    captured = {}

    def completion(messages):
        captured["messages"] = messages
        return '{"kind":"answer","answer":{"value":0,"unit":"unknown","observation_ids":[]}}'

    policy = JsonActionPolicy(completion)
    public = {"task_id": "t1", "question": "q", "as_of_time": "2025-03-31"}
    policy.reset(public)
    policy.act(AgentObservation(task=public, tool_schemas=[], tool_results=[], remaining_steps=3))
    rendered = json.dumps(captured["messages"])
    assert "oracle_steps" not in rendered
    assert "tolerance" not in rendered
    assert "get_financial_fact" in rendered


def test_harness_records_invalid_action_without_aborting(tmp_path, monkeypatch):
    from fintool_rl.database import build_fixture_snapshot
    from fintool_rl.tasks import generate_fixture_tasks

    db = build_fixture_snapshot(tmp_path / "snap.sqlite")
    task = generate_fixture_tasks(db)[0]

    class BadJsonPolicy:
        name = "BadJsonPolicy"
        privileged = False

        def reset(self, task):
            return None

        def act(self, observation):
            raise ActionParseError("model output is not one JSON object", raw_text="not-json <<<")

    trajectory, reward = HarnessRunner(db).run(task, BadJsonPolicy())
    assert trajectory.terminal_reason == "invalid_action"
    assert trajectory.actions[0]["kind"] == "invalid_action"
    assert "not-json" in trajectory.actions[0]["raw_text"]
    assert reward.hard_failure == "invalid_answer_format"
    assert reward.total == 0.0


def test_harness_records_model_call_error(tmp_path):
    from fintool_rl.database import build_fixture_snapshot
    from fintool_rl.tasks import generate_fixture_tasks

    db = build_fixture_snapshot(tmp_path / "snap.sqlite")
    task = generate_fixture_tasks(db)[0]

    class BoomApiPolicy:
        name = "BoomApiPolicy"
        privileged = False

        def reset(self, task):
            return None

        def act(self, observation):
            raise ModelCallError("HTTP 503: unavailable")

    trajectory, reward = HarnessRunner(db).run(task, BoomApiPolicy())
    assert trajectory.terminal_reason == "model_call_error"
    assert reward.hard_failure == "model_call_error"
    assert reward.total == 0.0


def test_guess_without_tools_still_execution_failure_when_answered():
    trajectory = Trajectory(
        trajectory_id="guess",
        task_id="t",
        policy_name="Guess",
        final_answer={"value": 1.0, "unit": "USD_million", "observation_ids": []},
        terminal_reason="answered",
    )
    from fintool_rl.contracts import TaskSpec

    task = TaskSpec(
        task_id="t",
        question="q",
        split="train",
        as_of_time="2025-03-31",
        difficulty="single_tool",
        template_family="financial_fact_lookup",
        answer={"value": 1.0, "unit": "USD_million", "tolerance": 0.0},
        oracle_steps=[{"id": "x"}],
        required_tool_families=["financial_statement"],
    )
    reward = grade_trajectory(task, trajectory)
    assert reward.hard_failure == "execution_failure"
