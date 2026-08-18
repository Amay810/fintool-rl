"""Fail-closed verification of the frozen evaluation set.

P1 measures a baseline and P5 re-measures after training.  If the two runs cover
different task sets the before/after comparison is meaningless, so these tests
pin the behaviour that makes a divergence loud: verification must *raise*, not
warn and not merely record.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fintool_rl.baseline_report import build_baseline_report
from fintool_rl.database import build_fixture_snapshot, file_sha256
from fintool_rl.evalset import (
    EvalsetMismatchError,
    build_evalset_manifest,
    load_evalset_manifest,
    split_task_ids_sha256,
    verify_evalset,
    verify_evalset_file,
    write_evalset_manifest,
)
from fintool_rl.harness import HarnessRunner, OraclePolicy, TrajectoryStore
from fintool_rl.tasks import generate_fixture_tasks, load_tasks, write_tasks

REPO_ROOT = Path(__file__).parents[1]
COMMITTED_FIXTURE_MANIFEST = REPO_ROOT / "data" / "evalset_manifest_fixture.json"


@pytest.fixture()
def frozen(tmp_path: Path):
    """A snapshot, its task file, and a manifest frozen from that exact file."""
    db = build_fixture_snapshot(tmp_path / "snapshot.sqlite")
    tasks = generate_fixture_tasks(db)
    tasks_path = tmp_path / "tasks.jsonl"
    write_tasks(tasks, tasks_path)
    manifest_path = tmp_path / "evalset_manifest.json"
    write_evalset_manifest(
        build_evalset_manifest(
            tasks, evalset_id="fixture-85-v1", tasks_path=tasks_path, db_path=db
        ),
        manifest_path,
    )
    return db, tasks, tasks_path, manifest_path


def test_matching_task_file_verifies(frozen) -> None:
    _, tasks, _, manifest_path = frozen
    result = verify_evalset(tasks, load_evalset_manifest(manifest_path))
    assert result["verified"] is True
    assert result["evalset_id"] == "fixture-85-v1"
    assert result["n_tasks"] == 85
    assert set(result["verified_splits"]) == {"train", "dev", "test", "challenge"}


def test_tampering_one_task_id_raises(frozen) -> None:
    _db, tasks, tasks_path, manifest_path = frozen
    victim = next(task for task in tasks if task.split == "test")
    original_id = victim.task_id
    victim.task_id = "task_tampered0000000"
    write_tasks(tasks, tasks_path)

    manifest = load_evalset_manifest(manifest_path)
    with pytest.raises(EvalsetMismatchError) as excinfo:
        verify_evalset(load_tasks(tasks_path), manifest)

    message = str(excinfo.value)
    # The error has to say which split and what the two digests were, or the
    # operator cannot tell a re-generated evalset from a wrong --tasks path.
    assert "'test'" in message
    assert manifest["splits"]["test"]["task_ids_sha256"] in message
    assert "fixture-85-v1" in message
    # Only the tampered split diverges; the untouched ones must not be implicated.
    for untouched in ("train", "dev", "challenge"):
        assert f"split {untouched!r}" not in message
    assert original_id != victim.task_id


def test_task_id_digest_survives_field_reordering_and_whitespace(frozen) -> None:
    """The design choice in `split_task_ids_sha256`, stated as a test.

    A whole-file hash is sensitive to key order and blank lines; the task id set
    is not.  This is why verification compares the split digests and keeps the
    file hash only as a record.
    """
    _, tasks, tasks_path, manifest_path = frozen
    manifest = load_evalset_manifest(manifest_path)
    original_file_sha = file_sha256(tasks_path)

    reordered = "\n\n".join(
        json.dumps(dict(reversed(list(task.to_dict().items()))), sort_keys=False)
        for task in tasks
    )
    tasks_path.write_text(reordered + "\n\n", encoding="utf-8")

    assert file_sha256(tasks_path) != original_file_sha
    assert file_sha256(tasks_path) != manifest["tasks_file_sha256"]
    # ...and yet the evalset is unchanged, so verification passes.
    result = verify_evalset(load_tasks(tasks_path), manifest)
    assert result["verified"] is True
    assert result["verified_splits"] == {
        split: digest["task_ids_sha256"] for split, digest in manifest["splits"].items()
    }


def test_explicit_override_does_not_raise(frozen) -> None:
    _, tasks, tasks_path, manifest_path = frozen
    tasks[0].task_id = "task_tampered0000000"
    write_tasks(tasks, tasks_path)

    with pytest.raises(EvalsetMismatchError):
        verify_evalset_file(load_tasks(tasks_path), manifest_path)

    allowed = verify_evalset_file(load_tasks(tasks_path), manifest_path, allow_mismatch=True)
    # The override must not be able to masquerade as a passed check.
    assert allowed["verified"] is False
    assert allowed["reason"] == "mismatch_explicitly_allowed"
    assert "task_ids_sha256" in allowed["mismatch"]


def test_absent_manifest_is_recorded_as_unverified(tmp_path: Path, frozen) -> None:
    _, tasks, _, _ = frozen
    result = verify_evalset_file(tasks, tmp_path / "nope.json")
    assert result["verified"] is False
    assert result["reason"] == "manifest_absent"


def _graded_report(tmp_path: Path, frozen, **kwargs):
    db, tasks, tasks_path, manifest_path = frozen
    subset = tasks[:5]
    store = TrajectoryStore(tmp_path / "store.sqlite")
    runner = HarnessRunner(db)
    for task in subset:
        trajectory, reward = runner.run(task, OraclePolicy())
        store.save(trajectory, reward)
    return build_baseline_report(
        tasks=subset,
        graded=store.load_graded(),
        db_path=db,
        tasks_path=tasks_path,
        store_path=tmp_path / "store.sqlite",
        policy_name="OraclePolicy",
        max_steps=8,
        evalset_manifest_path=manifest_path,
        **kwargs,
    )


def test_report_verifies_the_whole_task_file_not_the_graded_subset(tmp_path: Path, frozen) -> None:
    # Only 5 of 85 tasks are graded here.  Identity belongs to the frozen file, so a
    # run-time --split/--limit filter must not be mistaken for a changed evalset.
    report = _graded_report(tmp_path, frozen)
    evalset = report["protocol"]["evalset"]
    assert evalset["verified"] is True
    assert evalset["evalset_id"] == "fixture-85-v1"
    assert evalset["n_tasks"] == 85
    assert report["n_trajectories"] == 5
    manifest = load_evalset_manifest(frozen[3])
    assert evalset["verified_splits"] == {
        split: digest["task_ids_sha256"] for split, digest in manifest["splits"].items()
    }


def test_report_generation_fails_closed_on_a_changed_evalset(tmp_path: Path, frozen) -> None:
    _, tasks, tasks_path, _ = frozen
    tasks[0].task_id = "task_tampered0000000"
    write_tasks(tasks, tasks_path)
    with pytest.raises(EvalsetMismatchError):
        _graded_report(tmp_path, frozen)


def test_report_records_the_override_when_mismatch_is_allowed(tmp_path: Path, frozen) -> None:
    _, tasks, tasks_path, _ = frozen
    tasks[0].task_id = "task_tampered0000000"
    write_tasks(tasks, tasks_path)
    report = _graded_report(tmp_path, frozen, allow_evalset_mismatch=True)
    assert report["protocol"]["evalset"]["verified"] is False
    assert report["protocol"]["evalset"]["reason"] == "mismatch_explicitly_allowed"


def test_committed_fixture_manifest_still_matches_the_generator(tmp_path: Path) -> None:
    """The committed artifact must describe a task set that still exists.

    Guards against the fixture generator drifting away from the manifest checked
    into the repository, which would otherwise only surface at P5.
    """
    assert COMMITTED_FIXTURE_MANIFEST.exists(), (
        f"{COMMITTED_FIXTURE_MANIFEST} is missing — the frozen evalset must be committed, "
        "not regenerated per machine"
    )
    db = build_fixture_snapshot(tmp_path / "snapshot.sqlite")
    tasks = generate_fixture_tasks(db)
    manifest = load_evalset_manifest(COMMITTED_FIXTURE_MANIFEST)
    result = verify_evalset(tasks, manifest)
    assert result["verified"] is True
    assert manifest["snapshot_id"] == "synthetic-us-equities-v1"


def test_split_digest_is_order_independent() -> None:
    assert split_task_ids_sha256(["b", "a", "c"]) == split_task_ids_sha256(["a", "b", "c"])
    assert split_task_ids_sha256(["a", "b"]) != split_task_ids_sha256(["a", "b", "c"])
