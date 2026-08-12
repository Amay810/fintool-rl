"""Train-only headroom analysis and pre-registered RL readiness gates."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contracts import TaskSpec, Trajectory
from .reward import RewardVector


def terminal_success(reward: RewardVector) -> int:
    """Pure binary outcome used by routing and the main DrGRPO experiments."""
    return int(
        reward.hard_failure is None
        and reward.answer_correct == 1.0
        and reward.grounded == 1.0
    )


def graph_stratum(task: TaskSpec) -> str:
    depth = int(task.metadata.get("graph_depth", len(task.oracle_steps)))
    if depth <= 1:
        bucket = "depth_1"
    elif depth <= 3:
        bucket = "depth_2_3"
    elif depth <= 5:
        bucket = "depth_4_5"
    else:
        bucket = "depth_6_plus"
    return bucket


def graph_cell(task: TaskSpec) -> str:
    payload = {
        "family": task.template_family,
        "depth": int(task.metadata.get("graph_depth", len(task.oracle_steps))),
        "reuse": int(task.metadata.get("observation_reuse_count", 0)),
        "discovery": bool(task.metadata.get("discovery_required", False)),
        "distractors": int(task.metadata.get("distractor_count", 0)),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def pass_at_k(n: int, c: int, k: int) -> float:
    if not 0 <= c <= n or not 1 <= k <= n:
        raise ValueError("require 0 <= c <= n and 1 <= k <= n")
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def mixed_probability(p: float, group_size: int) -> float:
    return 1.0 - p**group_size - (1.0 - p) ** group_size


def expected_drgrpo_mass(p: float, group_size: int) -> float:
    return 2.0 * (group_size - 1) * p * (1.0 - p)


@dataclass(frozen=True)
class ReadinessThresholds:
    group_size: int = 8
    min_samples_per_task: int = 32
    min_band_tasks_per_stratum: int = 30
    min_band_tasks_total: int = 150
    opportunity_waste: float = 0.30
    icc_lower_bound: float = 0.10
    bootstrap_samples: int = 2000
    bootstrap_seed: int = 1729


def _binary_icc(rows: list[dict[str, Any]]) -> float:
    by_cell: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        by_cell[row["cell"]].append(int(row["success"]))
    observations = [value for values in by_cell.values() for value in values]
    if len(by_cell) < 2 or len(observations) < 2:
        return 0.0
    grand = sum(observations) / len(observations)
    between = sum(len(values) * ((sum(values) / len(values)) - grand) ** 2 for values in by_cell.values())
    within = sum(
        sum((value - sum(values) / len(values)) ** 2 for value in values)
        for values in by_cell.values()
    )
    between /= max(1, len(by_cell) - 1)
    within /= max(1, len(observations) - len(by_cell))
    mean_size = len(observations) / len(by_cell)
    denominator = between + (mean_size - 1.0) * within
    return 0.0 if denominator <= 0 else max(-1.0, min(1.0, (between - within) / denominator))


def _bootstrap_icc_lower(rows: list[dict[str, Any]], samples: int, seed: int) -> float:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[row["task_id"]].append(row)
    task_ids = sorted(by_task)
    if len(task_ids) < 2:
        return 0.0
    rng = random.Random(seed)
    estimates = []
    for _ in range(samples):
        resampled: list[dict[str, Any]] = []
        for index in range(len(task_ids)):
            chosen = rng.choice(task_ids)
            for row in by_task[chosen]:
                copied = dict(row)
                copied["task_id"] = f"{chosen}:{index}"
                resampled.append(copied)
        estimates.append(_binary_icc(resampled))
    estimates.sort()
    return estimates[max(0, int(0.025 * len(estimates)) - 1)]


def _leave_one_task_out_brier(rows: list[dict[str, Any]]) -> tuple[float, float]:
    total_success = sum(int(row["success"]) for row in rows)
    total_n = len(rows)
    cell_success: dict[str, int] = defaultdict(int)
    cell_n: dict[str, int] = defaultdict(int)
    task_success: dict[str, int] = defaultdict(int)
    task_n: dict[str, int] = defaultdict(int)
    task_cell: dict[str, str] = {}
    for row in rows:
        success = int(row["success"])
        cell = row["cell"]
        task_id = row["task_id"]
        cell_success[cell] += success
        cell_n[cell] += 1
        task_success[task_id] += success
        task_n[task_id] += 1
        task_cell[task_id] = cell
    global_loss = 0.0
    cell_loss = 0.0
    for task_id, n in task_n.items():
        held_success = task_success[task_id]
        global_p = (total_success - held_success + 1.0) / (total_n - n + 2.0)
        cell = task_cell[task_id]
        remaining_cell_n = cell_n[cell] - n
        if remaining_cell_n:
            cell_p = (cell_success[cell] - held_success + 1.0) / (remaining_cell_n + 2.0)
        else:
            cell_p = global_p
        global_loss += held_success * (1.0 - global_p) ** 2 + (n - held_success) * global_p**2
        cell_loss += held_success * (1.0 - cell_p) ** 2 + (n - held_success) * cell_p**2
    return global_loss / total_n, cell_loss / total_n


def analyze_readiness(
    tasks: Iterable[TaskSpec],
    graded: Iterable[tuple[Trajectory, RewardVector]],
    *,
    thresholds: ReadinessThresholds | None = None,
) -> dict[str, Any]:
    thresholds = thresholds or ReadinessThresholds()
    task_map = {task.task_id: task for task in tasks}
    if any(task.split != "train" for task in task_map.values()):
        raise ValueError("readiness analysis is train-only")
    rows: list[dict[str, Any]] = []
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trajectory, reward in graded:
        task = task_map.get(trajectory.task_id)
        if task is None:
            continue
        row = {
            "task_id": task.task_id,
            "cell": graph_cell(task),
            "stratum": graph_stratum(task),
            "success": terminal_success(reward),
            "generated_tokens": int(trajectory.generated_tokens),
            "policy_version": trajectory.policy_version,
        }
        rows.append(row)
        by_task[task.task_id].append(row)
    if not rows:
        raise ValueError("no matching train trajectories")
    missing_tokens = sum(row["generated_tokens"] <= 0 for row in rows)
    task_stats = []
    band_by_stratum: dict[str, int] = defaultdict(int)
    waste_numerator = 0.0
    waste_denominator = 0.0
    for task_id, task_rows in sorted(by_task.items()):
        n = len(task_rows)
        c = sum(row["success"] for row in task_rows)
        p = c / n
        task = task_map[task_id]
        stratum = graph_stratum(task)
        classification = "insufficient"
        if n >= thresholds.min_samples_per_task:
            if 4 <= c <= n - 4:
                classification = "core"
            elif 0 < c < n:
                classification = "frontier"
            else:
                classification = "apparent_degenerate"
        if classification in {"core", "frontier"}:
            band_by_stratum[stratum] += 1
        token_cost = sum(row["generated_tokens"] for row in task_rows)
        if token_cost > 0:
            waste_numerator += token_cost * (p**thresholds.group_size + (1 - p) ** thresholds.group_size)
            waste_denominator += token_cost
        task_stats.append({
            "task_id": task_id,
            "cell": graph_cell(task),
            "stratum": stratum,
            "n": n,
            "successes": c,
            "sampled_pass_at_1": p,
            "pass_at_8": pass_at_k(n, c, min(8, n)),
            "mixed_group_probability": mixed_probability(p, thresholds.group_size),
            "expected_drgrpo_mass": expected_drgrpo_mass(p, thresholds.group_size),
            "generated_tokens": token_cost,
            "classification": classification,
        })
    total_band = sum(band_by_stratum.values())
    claimed_strata = sorted({graph_stratum(task) for task in task_map.values()})
    learnability = (
        total_band >= thresholds.min_band_tasks_total
        and all(band_by_stratum[stratum] >= thresholds.min_band_tasks_per_stratum for stratum in claimed_strata)
    )
    uniform_waste = waste_numerator / waste_denominator if waste_denominator else None
    opportunity = uniform_waste is not None and uniform_waste >= thresholds.opportunity_waste
    icc = _binary_icc(rows)
    icc_lower = _bootstrap_icc_lower(
        rows, thresholds.bootstrap_samples, thresholds.bootstrap_seed
    )
    global_brier, cell_brier = _leave_one_task_out_brier(rows)
    cell_only = icc_lower > thresholds.icc_lower_bound and cell_brier < global_brier
    versions = sorted({row["policy_version"] for row in rows if row["policy_version"]})
    gates = {
        "token_accounting_complete": missing_tokens == 0,
        "single_policy_version": len(versions) == 1,
        "learnability": learnability,
        "curriculum_opportunity": opportunity,
        "cell_only_controller": cell_only,
    }
    return {
        "protocol": "m2.5-v1",
        "thresholds": thresholds.__dict__,
        "counts": {
            "tasks_expected": len(task_map),
            "tasks_observed": len(by_task),
            "trajectories": len(rows),
            "missing_token_counts": missing_tokens,
            "band_tasks_total": total_band,
            "band_tasks_by_stratum": dict(sorted(band_by_stratum.items())),
        },
        "policy_versions": versions,
        "uniform_token_weighted_degenerate_probability": uniform_waste,
        "controller_validation": {
            "icc": icc,
            "icc_bootstrap_lower_95": icc_lower,
            "global_loo_brier": global_brier,
            "cell_loo_brier": cell_brier,
            "selected": "cell_beta" if cell_only else "hierarchical_beta_binomial",
        },
        "gates": gates,
        "overall_go": all(gates[key] for key in (
            "token_accounting_complete", "single_policy_version", "learnability", "curriculum_opportunity"
        )),
        "tasks": task_stats,
    }


def write_readiness_report(report: dict[str, Any], path: Path | str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(report)
    encoded_tasks = json.dumps(payload.pop("tasks", []), sort_keys=True, separators=(",", ":"))
    payload["task_table_sha256"] = hashlib.sha256(encoded_tasks.encode()).hexdigest()
    payload["tasks"] = report.get("tasks", [])
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
