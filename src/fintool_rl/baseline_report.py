"""Frozen baseline reporting and failure aggregation."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .contracts import TaskSpec, Trajectory
from .database import snapshot_manifest
from .failure_taxonomy import SUCCESS_LABELS, FailureLabel, classify_failure, summarize_labels
from .reward import RewardVector
from .schema import FINANCIAL_METRICS, SCHEMA_BY_NAME


def file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def _slice_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0}
    return {
        "n": len(rows),
        "mean_reward": round(_mean([row["reward_total"] for row in rows]), 6),
        "answer_accuracy": round(_mean([row["answer_correct"] for row in rows]), 6),
        "grounded_rate": round(_mean([row["grounded"] for row in rows]), 6),
        "success_rate": round(_mean([1.0 if row["primary"] in SUCCESS_LABELS else 0.0 for row in rows]), 6),
        "hard_failures": dict(Counter(row["hard_failure"] or "none" for row in rows)),
        "primary_failures": dict(Counter(row["primary"] for row in rows)),
    }


def _contract_metrics(graded: list[tuple[Trajectory, RewardVector]]) -> dict[str, Any]:
    format_valid = 0
    observation_references = 0
    valid_observation_references = 0
    metric_arguments = 0
    valid_metric_arguments = 0
    for trajectory, reward in graded:
        format_valid += int(reward.format_valid == 1.0)
        known_observations: set[str] = set()
        for call in trajectory.tool_calls:
            schema = SCHEMA_BY_NAME.get(call.name)
            if schema:
                properties = schema["parameters"]["properties"]
                reference_fields = [name for name in properties if name.endswith("_observation_id")]
                for field in reference_fields:
                    observation_references += 1
                    value = call.arguments.get(field)
                    valid_observation_references += int(isinstance(value, str) and value in known_observations)
                if "metric" in properties:
                    metric_arguments += 1
                    valid_metric_arguments += int(call.arguments.get("metric") in FINANCIAL_METRICS)
            observation_id = (call.result.get("provenance") or {}).get("observation_id")
            if isinstance(observation_id, str):
                known_observations.add(observation_id)
    n = len(graded)
    return {
        "format_compliance_rate": format_valid / max(1, n),
        "format_compliant": format_valid,
        "n_trajectories": n,
        "observation_id_adherence_rate": (
            valid_observation_references / observation_references if observation_references else None
        ),
        "valid_observation_id_references": valid_observation_references,
        "observation_id_references": observation_references,
        "metric_naming_accuracy": valid_metric_arguments / metric_arguments if metric_arguments else None,
        "valid_metric_arguments": valid_metric_arguments,
        "metric_arguments": metric_arguments,
    }


def build_baseline_records(
    tasks: list[TaskSpec],
    graded: list[tuple[Trajectory, RewardVector]],
) -> list[dict[str, Any]]:
    task_by_id = {task.task_id: task for task in tasks}
    records: list[dict[str, Any]] = []
    for trajectory, reward in graded:
        task = task_by_id.get(trajectory.task_id)
        if task is None:
            continue
        label = classify_failure(task, trajectory, reward)
        records.append({
            "trajectory_id": trajectory.trajectory_id,
            "task_id": task.task_id,
            "split": task.split,
            "difficulty": task.difficulty,
            "oracle_step_count": len(task.oracle_steps),
            "template_family": task.template_family,
            "symbol": task.metadata.get("symbol"),
            "terminal_reason": trajectory.terminal_reason,
            "hard_failure": reward.hard_failure,
            "reward_total": reward.total,
            "answer_correct": reward.answer_correct,
            "grounded": reward.grounded,
            "format_valid": reward.format_valid,
            "efficiency": reward.efficiency,
            "required_family_coverage": reward.required_family_coverage,
            "primary": label.primary,
            "secondary": list(label.secondary),
            "evidence": label.evidence,
            "n_tool_calls": len(trajectory.tool_calls),
            "final_answer": trajectory.final_answer,
            "expected_answer": task.answer,
        })
    return records


def select_failure_examples(
    records: list[dict[str, Any]],
    *,
    per_label: int = 3,
) -> dict[str, list[dict[str, Any]]]:
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sorted(records, key=lambda item: item["task_id"]):
        label = row["primary"]
        if label in SUCCESS_LABELS:
            continue
        bucket = examples[label]
        if len(bucket) >= per_label:
            continue
        bucket.append({
            "task_id": row["task_id"],
            "split": row["split"],
            "template_family": row["template_family"],
            "symbol": row["symbol"],
            "terminal_reason": row["terminal_reason"],
            "hard_failure": row["hard_failure"],
            "final_answer": row["final_answer"],
            "expected_answer": row["expected_answer"],
            "secondary": row["secondary"],
            "evidence": row["evidence"],
        })
    return dict(sorted(examples.items()))


def build_baseline_report(
    *,
    tasks: list[TaskSpec],
    graded: list[tuple[Trajectory, RewardVector]],
    db_path: Path | str,
    tasks_path: Path | str,
    store_path: Path | str,
    policy_name: str,
    max_steps: int,
) -> dict[str, Any]:
    records = build_baseline_records(tasks, graded)
    label_objs = [
        FailureLabel(
            primary=row["primary"],
            secondary=tuple(row["secondary"]),
            evidence=row["evidence"],
        )
        for row in records
    ]
    by_split = {
        split: _slice_metrics([row for row in records if row["split"] == split])
        for split in sorted({row["split"] for row in records})
    }
    by_template = {
        family: _slice_metrics([row for row in records if row["template_family"] == family])
        for family in sorted({row["template_family"] for row in records})
    }
    by_difficulty = {
        difficulty: _slice_metrics([row for row in records if row["difficulty"] == difficulty])
        for difficulty in sorted({row["difficulty"] for row in records})
    }
    by_step_count = {
        str(step_count): _slice_metrics([row for row in records if row["oracle_step_count"] == step_count])
        for step_count in sorted({row["oracle_step_count"] for row in records})
    }
    manifest = snapshot_manifest(db_path)
    return {
        "protocol": {
            "name": "fintool-rl-m1-baseline",
            "policy_name": policy_name,
            "model_env": {
                "FINTOOL_LLM_BASE_URL": os.environ.get("FINTOOL_LLM_BASE_URL", ""),
                "FINTOOL_LLM_MODEL": os.environ.get("FINTOOL_LLM_MODEL", ""),
            },
            "max_steps": max_steps,
            "decoding": {
                "temperature": float(os.environ.get("FINTOOL_LLM_TEMPERATURE", "0.7")),
                "top_p": float(os.environ.get("FINTOOL_LLM_TOP_P", "0.8")),
                "top_k": int(os.environ.get("FINTOOL_LLM_TOP_K", "20")),
                "min_p": float(os.environ.get("FINTOOL_LLM_MIN_P", "0")),
                "seed": int(os.environ.get("FINTOOL_LLM_SEED", "0")),
                "tool_protocol": "openai-native-function-calling",
            },
            "db_path": str(db_path),
            "tasks_path": str(tasks_path),
            "store_path": str(store_path),
            "tasks_sha256": file_sha256(tasks_path),
            "snapshot_sha256": manifest["sha256"],
            "snapshot_id": manifest["metadata"].get("snapshot_id"),
            "as_of_time": manifest["metadata"].get("as_of_time"),
        },
        "overall": _slice_metrics(records),
        "contract_metrics": _contract_metrics(graded),
        "by_split": by_split,
        "by_template": by_template,
        "by_difficulty": by_difficulty,
        "by_step_count": by_step_count,
        "failure_taxonomy": summarize_labels(label_objs),
        "failure_examples": select_failure_examples(records),
        "n_trajectories": len(records),
        "n_tasks_available": len(tasks),
    }


def write_report(report: dict[str, Any], path: Path | str) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def write_failure_table(records: list[dict[str, Any]], path: Path | str) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = [json.dumps(row, ensure_ascii=False, sort_keys=True) for row in records]
    output.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    return output
