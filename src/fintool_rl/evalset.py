"""Frozen evaluation-set identity and fail-closed verification.

P1 measures a baseline and P5 re-measures after training.  A before/after
comparison is only meaningful if both runs covered the *same* task set, so the
identity of that set has to be pinned to a committed artifact and checked, not
merely recorded.

`baseline_report.py` already writes `tasks_sha256` into every report, but a
recorded hash is fail-open: pointing a run at the wrong task file produces a
report that faithfully records the wrong hash and raises nothing.  This module
supplies the missing half — a manifest that is committed to the repository, and
a verification step that raises.

Identity is defined by the **set of task ids per split**, not by the bytes of the
task file.  A whole-file hash is sensitive to key order, whitespace, and changes
to fields that have nothing to do with which questions are asked; the task id set
is what "this evaluation covered these questions" actually means.  Both digests
are recorded, and the split digests are what verification compares.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .contracts import TaskSpec
from .database import file_sha256, snapshot_manifest

DEFAULT_MANIFEST_PATH = Path("data/evalset_manifest.json")


class EvalsetMismatchError(RuntimeError):
    """The task set on disk is not the frozen evaluation set."""


def split_task_ids_sha256(task_ids: Iterable[str]) -> str:
    """Digest of a split's task ids: sorted, newline-joined, UTF-8, sha256.

    Order-independent and insensitive to anything about a task except its id.
    """
    joined = "\n".join(sorted(task_ids))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def split_digests(tasks: Iterable[TaskSpec]) -> dict[str, dict[str, Any]]:
    by_split: dict[str, list[str]] = {}
    for task in tasks:
        by_split.setdefault(task.split, []).append(task.task_id)
    return {
        split: {"n_tasks": len(ids), "task_ids_sha256": split_task_ids_sha256(ids)}
        for split, ids in sorted(by_split.items())
    }


def build_evalset_manifest(
    tasks: list[TaskSpec],
    *,
    evalset_id: str,
    tasks_path: Path | str,
    db_path: Path | str,
) -> dict[str, Any]:
    if not evalset_id:
        raise ValueError("evalset_id must be a non-empty human-readable identifier")
    if not tasks:
        raise ValueError("refusing to freeze an empty task set")
    snapshot = snapshot_manifest(db_path)
    # `generator` is written into task metadata by tasks.py; it is recorded here as
    # observed rather than as a newly minted version constant.
    generators = sorted({str(task.metadata.get("generator", "unknown")) for task in tasks})
    return {
        "evalset_id": evalset_id,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_tasks": len(tasks),
        "generators": generators,
        "splits": split_digests(tasks),
        "tasks_file": str(tasks_path),
        "tasks_file_sha256": file_sha256(tasks_path),
        "snapshot_id": snapshot["metadata"].get("snapshot_id"),
        "snapshot_sha256": snapshot["sha256"],
        "as_of_time": snapshot["metadata"].get("as_of_time"),
    }


def write_evalset_manifest(manifest: dict[str, Any], path: Path | str) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output


def load_evalset_manifest(path: Path | str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify_evalset(tasks: list[TaskSpec], manifest: dict[str, Any]) -> dict[str, Any]:
    """Raise `EvalsetMismatchError` unless `tasks` is the frozen evaluation set.

    Returns the verified identity to be recorded in a report's protocol block.
    The caller must verify *before* writing that block: recording first and
    comparing afterwards would only restate what the run already chose to do.
    """
    expected_splits = manifest.get("splits") or {}
    actual_splits = split_digests(tasks)

    problems: list[str] = []
    for split in sorted(set(expected_splits) | set(actual_splits)):
        expected = expected_splits.get(split)
        actual = actual_splits.get(split)
        if expected is None:
            problems.append(
                f"split {split!r}: absent from the frozen manifest, but the task file "
                f"contains {actual['n_tasks']} such tasks "
                f"(actual task_ids_sha256={actual['task_ids_sha256']})"
            )
            continue
        if actual is None:
            problems.append(
                f"split {split!r}: frozen manifest expects {expected['n_tasks']} tasks "
                f"(expected task_ids_sha256={expected['task_ids_sha256']}), "
                f"but the task file contains none"
            )
            continue
        if expected.get("n_tasks") != actual["n_tasks"]:
            problems.append(
                f"split {split!r}: n_tasks expected={expected.get('n_tasks')} "
                f"actual={actual['n_tasks']}"
            )
        if expected.get("task_ids_sha256") != actual["task_ids_sha256"]:
            problems.append(
                f"split {split!r}: task_ids_sha256 "
                f"expected={expected.get('task_ids_sha256')} "
                f"actual={actual['task_ids_sha256']}"
            )

    if problems:
        raise EvalsetMismatchError(
            f"task set does not match frozen evalset {manifest.get('evalset_id')!r}; "
            "before/after comparisons against this evalset would be invalid. "
            + " | ".join(problems)
        )

    return {
        "evalset_id": manifest.get("evalset_id"),
        "verified": True,
        "verified_splits": {
            split: digest["task_ids_sha256"] for split, digest in actual_splits.items()
        },
        "n_tasks": sum(digest["n_tasks"] for digest in actual_splits.values()),
        "manifest_tasks_file_sha256": manifest.get("tasks_file_sha256"),
        "manifest_snapshot_sha256": manifest.get("snapshot_sha256"),
    }


def verify_evalset_file(
    tasks: list[TaskSpec],
    manifest_path: Path | str | None,
    *,
    allow_mismatch: bool = False,
) -> dict[str, Any]:
    """Verify against a manifest file if one exists; describe the outcome either way.

    A report whose evalset block says `verified: false` is self-evidently
    unverified, which is the point: absence of a manifest must not look the same
    as a passed check.
    """
    if manifest_path is None:
        return {"verified": False, "reason": "verification_disabled"}
    path = Path(manifest_path)
    if not path.exists():
        return {"verified": False, "reason": "manifest_absent", "manifest_path": str(path)}
    manifest = load_evalset_manifest(path)
    try:
        result = verify_evalset(tasks, manifest)
    except EvalsetMismatchError as exc:
        if not allow_mismatch:
            raise
        return {
            "verified": False,
            "reason": "mismatch_explicitly_allowed",
            "manifest_path": str(path),
            "evalset_id": manifest.get("evalset_id"),
            "mismatch": str(exc),
        }
    result["manifest_path"] = str(path)
    return result
