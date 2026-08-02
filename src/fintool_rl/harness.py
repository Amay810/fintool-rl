"""Leakage-resistant task runner and immutable trajectory store."""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .contracts import AgentAction, TaskSpec, Trajectory
from .oracle import resolve_references
from .reward import RewardVector, grade_trajectory
from .schema import TOOL_SCHEMAS
from .tools import FinancialTools


@dataclass(frozen=True)
class AgentObservation:
    task: dict[str, Any]
    tool_schemas: list[dict[str, Any]]
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    remaining_steps: int = 0


class Policy(Protocol):
    name: str
    privileged: bool

    def reset(self, task: TaskSpec | dict[str, Any]) -> None: ...

    def act(self, observation: AgentObservation) -> AgentAction: ...


class OraclePolicy:
    """Privileged executable oracle used only for environment smoke tests."""

    name = "OraclePolicy"
    privileged = True

    def __init__(self) -> None:
        self.task: TaskSpec | None = None
        self.step_index = 0
        self.step_results: dict[str, dict[str, Any]] = {}

    def reset(self, task: TaskSpec | dict[str, Any]) -> None:
        if not isinstance(task, TaskSpec):
            raise TypeError("OraclePolicy requires the hidden TaskSpec")
        self.task = task
        self.step_index = 0
        self.step_results = {}

    def act(self, observation: AgentObservation) -> AgentAction:
        assert self.task is not None
        if self.step_index > 0 and len(observation.tool_results) >= self.step_index:
            previous_step = self.task.oracle_steps[self.step_index - 1]
            self.step_results[previous_step["id"]] = observation.tool_results[self.step_index - 1]
        if self.step_index < len(self.task.oracle_steps):
            step = self.task.oracle_steps[self.step_index]
            arguments = resolve_references(step["arguments"], self.step_results)
            self.step_index += 1
            return AgentAction.tool(step["tool"], **arguments)
        final_step = self.task.oracle_steps[-1]
        final = self.step_results[final_step["id"]]
        return AgentAction.final(
            value=final["scalar"],
            unit=final["unit"],
            observation_ids=[final["provenance"]["observation_id"]],
        )


class ReplayPolicy:
    """Replay a frozen action sequence without access to hidden task fields."""

    name = "ReplayPolicy"
    privileged = False

    def __init__(self, actions: list[dict[str, Any]]):
        self.actions = actions
        self.index = 0

    def reset(self, task: TaskSpec | dict[str, Any]) -> None:
        self.index = 0

    def act(self, observation: AgentObservation) -> AgentAction:
        if self.index >= len(self.actions):
            return AgentAction(kind="answer", answer=None)
        action = AgentAction(**self.actions[self.index])
        self.index += 1
        return action


class HarnessRunner:
    def __init__(self, db_path: Path | str, *, max_steps: int = 8):
        self.db_path = Path(db_path)
        self.max_steps = max_steps

    def run(self, task: TaskSpec, policy: Policy) -> tuple[Trajectory, RewardVector]:
        from .policies import ActionParseError, ModelCallError

        policy.reset(task if policy.privileged else task.public_view())
        tools = FinancialTools(self.db_path)
        trajectory = Trajectory(
            trajectory_id=f"traj_{uuid.uuid4().hex}",
            task_id=task.task_id,
            policy_name=policy.name,
        )
        tool_results: list[dict[str, Any]] = []
        for step_index in range(self.max_steps + 1):
            observation = AgentObservation(
                task=task.public_view(),
                tool_schemas=TOOL_SCHEMAS,
                tool_results=list(tool_results),
                remaining_steps=max(0, self.max_steps - step_index),
            )
            try:
                action = policy.act(observation)
            except ActionParseError as exc:
                trajectory.actions.append({
                    "kind": "invalid_action",
                    "error": str(exc),
                    "raw_text": (exc.raw_text or "")[:2000],
                })
                trajectory.terminal_reason = "invalid_action"
                break
            except ModelCallError as exc:
                trajectory.actions.append({
                    "kind": "model_call_error",
                    "error": str(exc)[:2000],
                })
                trajectory.terminal_reason = "model_call_error"
                break
            trajectory.actions.append(asdict(action))
            if action.kind == "answer":
                trajectory.final_answer = action.answer
                trajectory.terminal_reason = "answered"
                break
            if action.kind != "tool" or not action.tool_name:
                trajectory.terminal_reason = "invalid_action"
                break
            result = tools.call(action.tool_name, **action.arguments)
            tool_results.append(result)
        else:
            trajectory.terminal_reason = "max_steps"
        trajectory.tool_calls = list(tools.calls)
        return trajectory, grade_trajectory(task, trajectory)


class TrajectoryStore:
    """Append-only SQLite store for trajectories and their reward vectors."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path)
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS trajectories (
                    trajectory_id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL,
                    policy_name TEXT NOT NULL,
                    trajectory_json TEXT NOT NULL,
                    reward_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS trajectories_task_id ON trajectories(task_id)"
            )
            conn.commit()
        finally:
            conn.close()

    def save(self, trajectory: Trajectory, reward: RewardVector) -> None:
        conn = sqlite3.connect(self.path)
        try:
            conn.execute(
                "INSERT INTO trajectories VALUES (?, ?, ?, ?, ?)",
                (
                    trajectory.trajectory_id,
                    trajectory.task_id,
                    trajectory.policy_name,
                    json.dumps(trajectory.to_dict(), ensure_ascii=False, sort_keys=True),
                    json.dumps(reward.to_dict(), ensure_ascii=False, sort_keys=True),
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def task_ids(self) -> set[str]:
        conn = sqlite3.connect(self.path)
        try:
            rows = conn.execute("SELECT DISTINCT task_id FROM trajectories").fetchall()
        finally:
            conn.close()
        return {row[0] for row in rows}

    def rows(self) -> list[dict[str, Any]]:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        try:
            return [dict(row) for row in conn.execute("SELECT * FROM trajectories ORDER BY trajectory_id")]
        finally:
            conn.close()

    def load_graded(self) -> list[tuple[Trajectory, RewardVector]]:
        from .contracts import ToolCall
        from .reward import RewardVector as RewardVectorType

        graded: list[tuple[Trajectory, RewardVector]] = []
        for row in self.rows():
            payload = json.loads(row["trajectory_json"])
            tool_calls = [ToolCall(**call) for call in payload.get("tool_calls", [])]
            trajectory = Trajectory(
                trajectory_id=payload["trajectory_id"],
                task_id=payload["task_id"],
                policy_name=payload["policy_name"],
                actions=payload.get("actions", []),
                tool_calls=tool_calls,
                final_answer=payload.get("final_answer"),
                terminal_reason=payload.get("terminal_reason"),
            )
            reward = RewardVectorType(**json.loads(row["reward_json"]))
            graded.append((trajectory, reward))
        return graded
