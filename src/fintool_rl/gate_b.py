"""Gate B: stochastic action-reachability analysis.

This module grades already-collected K-sample groups. It does not call a model.
Pass@K here means at least one sample is numerically correct without a hard failure.
Positive@K uses the separate positive-v1 selection contract.
"""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from typing import Any

from .contracts import Trajectory
from .reward import RewardVector
from .schema import TOOL_FAMILY_BY_NAME
from .selection import select_positive_trajectory

GATE_B_VERSION = "gate-b-v1"


def _canonical(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def action_fingerprint(action: dict[str, Any]) -> str:
    if action.get("kind") == "tool":
        return _canonical({
            "kind": "tool",
            "tool_name": action.get("tool_name"),
            "arguments": action.get("arguments") or {},
        })
    if action.get("kind") == "answer":
        return _canonical({"kind": "answer", "answer": action.get("answer")})
    return _canonical(action)


def trajectory_fingerprint(trajectory: Trajectory) -> str:
    return _canonical({
        "actions": [action_fingerprint(action) for action in trajectory.actions],
        "terminal_reason": trajectory.terminal_reason,
    })


def calculator_called(trajectory: Trajectory) -> bool:
    return any(TOOL_FAMILY_BY_NAME.get(call.name) == "calculator" for call in trajectory.tool_calls)


def _sample_variance(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return statistics.pvariance(values)


def summarize_task_group(
    task_id: str,
    graded: list[tuple[Trajectory, RewardVector]],
) -> dict[str, Any]:
    if not graded:
        raise ValueError(f"no trajectories for task {task_id}")
    totals = [reward.total for _, reward in graded]
    correct = [
        reward.answer_correct == 1.0 and reward.hard_failure is None
        for _, reward in graded
    ]
    positive = [select_positive_trajectory(reward).selected for _, reward in graded]
    action_fps = [action_fingerprint(action) for trajectory, _ in graded for action in trajectory.actions]
    traj_fps = [trajectory_fingerprint(trajectory) for trajectory, _ in graded]
    n_actions = len(action_fps)
    return {
        "task_id": task_id,
        "k": len(graded),
        "pass_at_k": int(any(correct)),
        "n_correct": int(sum(correct)),
        "positive_at_k": int(any(positive)),
        "n_positive": int(sum(positive)),
        "calculator_called_rate": sum(calculator_called(trajectory) for trajectory, _ in graded) / len(graded),
        "unique_trajectories": len(set(traj_fps)),
        "trajectory_diversity": len(set(traj_fps)) / len(graded),
        "unique_actions": len(set(action_fps)),
        "action_diversity": (len(set(action_fps)) / n_actions) if n_actions else 0.0,
        "reward_mean": sum(totals) / len(totals),
        "reward_variance": _sample_variance(totals),
    }


def build_reachability_report(
    graded: list[tuple[Trajectory, RewardVector]],
    *,
    temperature: float,
    k: int,
    policy_name: str,
) -> dict[str, Any]:
    grouped: dict[str, list[tuple[Trajectory, RewardVector]]] = defaultdict(list)
    for trajectory, reward in graded:
        grouped[trajectory.task_id].append((trajectory, reward))
    per_task = [summarize_task_group(task_id, rows) for task_id, rows in sorted(grouped.items())]
    n_tasks = len(per_task)
    if n_tasks == 0:
        raise ValueError("no graded trajectories")
    return {
        "gate": GATE_B_VERSION,
        "temperature": temperature,
        "k": k,
        "policy_name": policy_name,
        "n_tasks": n_tasks,
        "n_trajectories": len(graded),
        "pass_at_k": sum(row["pass_at_k"] for row in per_task) / n_tasks,
        "positive_at_k": sum(row["positive_at_k"] for row in per_task) / n_tasks,
        "mean_calculator_called_rate": sum(row["calculator_called_rate"] for row in per_task) / n_tasks,
        "mean_trajectory_diversity": sum(row["trajectory_diversity"] for row in per_task) / n_tasks,
        "mean_action_diversity": sum(row["action_diversity"] for row in per_task) / n_tasks,
        "mean_reward_variance": sum(row["reward_variance"] for row in per_task) / n_tasks,
        "tasks_with_multiple_trajectories": sum(row["unique_trajectories"] > 1 for row in per_task),
        "per_task": per_task,
    }


def require_gate_b_settings(*, temperature: float, k: int) -> None:
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Gate B requires temperature>0")
    if k <= 1:
        raise ValueError("Gate B requires K>1")
