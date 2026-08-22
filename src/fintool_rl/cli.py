"""Command-line entry points for the M1 environment."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .baseline_report import (
    build_baseline_records,
    build_baseline_report,
    write_failure_table,
    write_report,
)
from .database import build_fixture_snapshot, file_sha256, snapshot_manifest, write_manifest
from .evalset import (
    DEFAULT_MANIFEST_PATH,
    build_evalset_manifest,
    verify_evalset_file,
    write_evalset_manifest,
)
from .gate_b import build_reachability_report, require_gate_b_settings
from .harness import HarnessRunner, OraclePolicy, TrajectoryStore
from .policies import OpenAICompatiblePolicy
from .sec import (
    build_sec_snapshot,
    download_companyfacts,
    download_ticker_exchange,
    load_company_mapping,
    resolve_universe,
)
from .tasks import (
    assert_no_fact_leakage,
    generate_fixture_tasks,
    generate_snapshot_tasks,
    load_tasks,
    select_split_targets,
    write_tasks,
)


def bootstrap(args: argparse.Namespace) -> None:
    db_path = Path(args.db)
    task_path = Path(args.tasks)
    manifest_path = Path(args.manifest)
    build_fixture_snapshot(db_path, overwrite=args.overwrite)
    tasks = generate_fixture_tasks(db_path)
    assert_no_fact_leakage(tasks)
    write_tasks(tasks, task_path)
    write_manifest(db_path, manifest_path)
    print(json.dumps({
        "snapshot": snapshot_manifest(db_path),
        "tasks": len(tasks),
        "splits": dict(Counter(task.split for task in tasks)),
        "task_path": str(task_path),
    }, indent=2, sort_keys=True))


def smoke(args: argparse.Namespace) -> None:
    tasks = load_tasks(args.tasks)
    if args.limit:
        tasks = tasks[: args.limit]
    runner = HarnessRunner(args.db)
    store = TrajectoryStore(args.store)
    rewards = []
    for task in tasks:
        trajectory, reward = runner.run(task, OraclePolicy())
        store.save(trajectory, reward)
        rewards.append(reward)
    report = {
        "n": len(rewards),
        "success": sum(reward.total == 1.0 for reward in rewards),
        "mean_reward": sum(reward.total for reward in rewards) / max(1, len(rewards)),
        "hard_failures": dict(Counter(reward.hard_failure or "none" for reward in rewards)),
        "store": str(args.store),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


def _filter_tasks(args: argparse.Namespace):
    tasks = load_tasks(args.tasks)
    if args.split:
        tasks = [task for task in tasks if task.split == args.split]
    if getattr(args, "template_family", ""):
        tasks = [task for task in tasks if task.template_family == args.template_family]
    if getattr(args, "difficulty", ""):
        tasks = [task for task in tasks if task.difficulty == args.difficulty]
    if args.limit:
        tasks = tasks[: args.limit]
    return tasks


def baseline(args: argparse.Namespace) -> None:
    tasks = _filter_tasks(args)
    store = TrajectoryStore(args.store)
    done = store.task_ids() if args.skip_existing else set()
    pending = [task for task in tasks if task.task_id not in done]
    policy = OpenAICompatiblePolicy.from_env()
    runner = HarnessRunner(args.db, max_steps=args.max_steps)
    newly_graded = 0
    for index, task in enumerate(pending, start=1):
        trajectory, reward = runner.run(task, policy)
        store.save(trajectory, reward)
        newly_graded += 1
        if args.progress_every and index % args.progress_every == 0:
            print(json.dumps({
                "progress": index,
                "pending": len(pending),
                "task_id": task.task_id,
                "reward_total": reward.total,
                "hard_failure": reward.hard_failure,
                "terminal_reason": trajectory.terminal_reason,
            }, sort_keys=True), flush=True)

    graded = store.load_graded()
    # Keep only trajectories for the requested task filter when analyzing.
    selected_ids = {task.task_id for task in tasks}
    graded = [(trajectory, reward) for trajectory, reward in graded if trajectory.task_id in selected_ids]
    report = build_baseline_report(
        tasks=tasks,
        graded=graded,
        db_path=args.db,
        tasks_path=args.tasks,
        store_path=args.store,
        policy_name=policy.name,
        max_steps=args.max_steps,
        evalset_manifest_path=args.evalset_manifest,
        allow_evalset_mismatch=args.allow_evalset_mismatch,
    )
    report["run"] = {
        "requested_tasks": len(tasks),
        "already_done": len(tasks) - len(pending),
        "newly_graded": newly_graded,
        "skipped_existing": bool(args.skip_existing),
    }
    if args.report:
        write_report(report, args.report)
    if args.failure_table:
        records = build_baseline_records(tasks, graded)
        write_failure_table(records, args.failure_table)
    print(json.dumps({
        "policy": policy.name,
        "requested_tasks": len(tasks),
        "newly_graded": newly_graded,
        "n_trajectories": report["n_trajectories"],
        "overall": report["overall"],
        "failure_taxonomy": report["failure_taxonomy"]["primary"],
        "report": args.report or None,
        "store": str(args.store),
    }, indent=2, sort_keys=True))


def analyze_baseline(args: argparse.Namespace) -> None:
    tasks = _filter_tasks(args)
    store = TrajectoryStore(args.store)
    graded = [
        (trajectory, reward)
        for trajectory, reward in store.load_graded()
        if trajectory.task_id in {task.task_id for task in tasks}
    ]
    if not graded:
        raise SystemExit(f"no trajectories found in {args.store}")
    policy_name = graded[0][0].policy_name
    report = build_baseline_report(
        tasks=tasks,
        graded=graded,
        db_path=args.db,
        tasks_path=args.tasks,
        store_path=args.store,
        policy_name=policy_name,
        max_steps=args.max_steps,
        evalset_manifest_path=args.evalset_manifest,
        allow_evalset_mismatch=args.allow_evalset_mismatch,
    )
    write_report(report, args.report)
    if args.failure_table:
        write_failure_table(build_baseline_records(tasks, graded), args.failure_table)
    print(json.dumps({
        "policy": policy_name,
        "n_trajectories": report["n_trajectories"],
        "overall": report["overall"],
        "failure_taxonomy": report["failure_taxonomy"]["primary"],
        "report": args.report,
    }, indent=2, sort_keys=True))


def sec_download(args: argparse.Namespace) -> None:
    mapping = load_company_mapping(args.mapping)
    download_companyfacts(mapping, args.output_dir, user_agent=args.user_agent)
    print(json.dumps({"downloaded": len(mapping), "output_dir": args.output_dir}, indent=2))


def sec_import(args: argparse.Namespace) -> None:
    path = build_sec_snapshot(
        args.mapping,
        args.input_dir,
        args.db,
        as_of_time=args.as_of_time,
        overwrite=args.overwrite,
    )
    write_manifest(path, args.manifest)
    print(json.dumps(snapshot_manifest(path), indent=2, sort_keys=True))


def generate_tasks(args: argparse.Namespace) -> None:
    mapping = load_company_mapping(args.mapping)
    split_map = {str(row["symbol"]).upper(): row.get("split", "train") for row in mapping}
    tasks = generate_snapshot_tasks(args.db, split_map, recent_years=args.recent_years)
    targets = {
        split: target
        for split, target in {
            "train": args.train_target,
            "dev": args.dev_target,
            "test": args.test_target,
        }.items()
        if target is not None
    }
    if targets:
        tasks = select_split_targets(tasks, targets)
    write_tasks(tasks, args.output)
    print(json.dumps({
        "tasks": len(tasks),
        "splits": dict(Counter(task.split for task in tasks)),
        "templates": dict(Counter(task.template_family for task in tasks)),
        "output": args.output,
    }, indent=2, sort_keys=True))


def check_runtime(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root)
    report: dict[str, Any] = {"data_root": str(data_root), "checks": []}

    def record(name: str, ok: bool, **detail: object) -> None:
        report["checks"].append({"name": name, "ok": ok, **detail})

    if not str(data_root.resolve()).startswith("/root/autodl-tmp"):
        record("data_root_on_data_disk", False, detail=str(data_root))
    else:
        record("data_root_on_data_disk", True)

    db = Path(args.db)
    record("snapshot_exists", db.is_file(), path=str(db))
    if db.is_file():
        digest = file_sha256(db)
        expected = snapshot_manifest(db).get("sha256")
        record("snapshot_sha256", digest == expected, sha256=digest, expected=expected)

    tasks = load_tasks(args.tasks)
    record("tasks_loaded", bool(tasks), n=len(tasks))
    verification = verify_evalset_file(tasks, args.evalset_manifest)
    record("evalset_verified", bool(verification.get("verified")), **verification)

    model_dir = Path(args.model_dir)
    weight_files = list(model_dir.glob("*.safetensors")) + list(model_dir.glob("*.bin"))
    record(
        "model_weights_present",
        model_dir.is_dir() and bool(weight_files),
        path=str(model_dir),
        n_weight_files=len(weight_files),
    )
    tokenizer_ok = (model_dir / "tokenizer.json").is_file() or (model_dir / "tokenizer_config.json").is_file()
    record("tokenizer_files_present", tokenizer_ok, path=str(model_dir))

    if args.load_tokenizer:
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
            encoded = tokenizer("Gate B runtime check")
            record("tokenizer_loads", True, n_tokens=len(encoded["input_ids"]))
        except Exception as exc:  # import or 2GiB OOM must be named, not hidden
            record("tokenizer_loads", False, error=type(exc).__name__, detail=str(exc)[:300])

    skip_fail = set()
    if not args.require_model:
        skip_fail.update({"model_weights_present", "tokenizer_files_present", "tokenizer_loads"})
    failed = [row["name"] for row in report["checks"] if not row["ok"] and row["name"] not in skip_fail]
    report["ok"] = not failed
    report["failed"] = failed
    print(json.dumps(report, indent=2, sort_keys=True))
    if failed:
        raise SystemExit(f"runtime checks failed: {failed}")


def probe_reachability(args: argparse.Namespace) -> None:
    require_gate_b_settings(temperature=args.temperature, k=args.k)
    tasks = _filter_tasks(args)
    store = TrajectoryStore(args.store)
    existing = [trajectory for trajectory, _ in store.load_graded()]
    done_counts: dict[str, int] = {}
    for trajectory in existing:
        done_counts[trajectory.task_id] = done_counts.get(trajectory.task_id, 0) + 1

    policy = OpenAICompatiblePolicy.from_env(temperature=args.temperature)
    runner = HarnessRunner(args.db, max_steps=args.max_steps)
    newly_graded = 0
    for task in tasks:
        already = done_counts.get(task.task_id, 0)
        for _ in range(max(0, args.k - already)):
            trajectory, reward = runner.run(task, policy)
            store.save(trajectory, reward)
            newly_graded += 1

    selected_ids = {task.task_id for task in tasks}
    graded = [
        (trajectory, reward)
        for trajectory, reward in store.load_graded()
        if trajectory.task_id in selected_ids
    ]
    report = build_reachability_report(
        graded,
        temperature=args.temperature,
        k=args.k,
        policy_name=policy.name,
    )
    report["run"] = {
        "requested_tasks": len(tasks),
        "newly_graded": newly_graded,
        "store": str(args.store),
    }
    if args.report:
        write_report(report, args.report)
    print(json.dumps({
        "gate": report["gate"],
        "n_tasks": report["n_tasks"],
        "n_trajectories": report["n_trajectories"],
        "pass_at_k": report["pass_at_k"],
        "mean_trajectory_diversity": report["mean_trajectory_diversity"],
        "mean_calculator_called_rate": report["mean_calculator_called_rate"],
        "mean_reward_variance": report["mean_reward_variance"],
        "report": args.report or None,
    }, indent=2, sort_keys=True))


def freeze_evalset(args: argparse.Namespace) -> None:
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise SystemExit(
            f"{output} already exists; refusing to re-freeze an evaluation set without "
            "--overwrite (silently re-freezing is exactly what this manifest prevents)"
        )
    tasks = load_tasks(args.tasks)
    manifest = build_evalset_manifest(
        tasks,
        evalset_id=args.evalset_id,
        tasks_path=args.tasks,
        db_path=args.db,
    )
    write_evalset_manifest(manifest, output)
    print(json.dumps({
        "evalset_id": manifest["evalset_id"],
        "n_tasks": manifest["n_tasks"],
        "splits": {split: digest["n_tasks"] for split, digest in manifest["splits"].items()},
        "output": str(output),
        "committed": "this file must be committed to git; it is the frozen identity",
    }, indent=2, sort_keys=True))


def sec_resolve(args: argparse.Namespace) -> None:
    download_ticker_exchange(args.ticker_file, user_agent=args.user_agent)
    mapping = resolve_universe(args.universe, args.ticker_file)
    Path(args.output).write_text(json.dumps(mapping, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"companies": len(mapping), "output": args.output}, indent=2))


def _add_evalset_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--evalset-manifest",
        default=str(DEFAULT_MANIFEST_PATH),
        help="Frozen evaluation-set manifest to verify against. Verification is skipped "
             "only when the file does not exist.",
    )
    parser.add_argument(
        "--allow-evalset-mismatch",
        action="store_true",
        help="Proceed despite a task set that does not match the frozen manifest. "
             "For knowing human override only; before/after comparisons produced this "
             "way are not comparable.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    boot = subparsers.add_parser("bootstrap", help="Build the synthetic snapshot and generated tasks.")
    boot.add_argument("--db", default="data/fixture_snapshot.sqlite")
    boot.add_argument("--tasks", default="data/generated_fixture_tasks.jsonl")
    boot.add_argument("--manifest", default="data/fixture_snapshot.manifest.json")
    boot.add_argument("--overwrite", action="store_true")
    boot.set_defaults(func=bootstrap)

    run = subparsers.add_parser("smoke", help="Run privileged executable oracles through the harness.")
    run.add_argument("--db", default="data/fixture_snapshot.sqlite")
    run.add_argument("--tasks", default="data/generated_fixture_tasks.jsonl")
    run.add_argument("--store", default="logs/oracle_smoke.sqlite")
    run.add_argument("--limit", type=int, default=0)
    run.set_defaults(func=smoke)

    base = subparsers.add_parser("baseline", help="Run an OpenAI-compatible model baseline.")
    base.add_argument("--db", default="data/sec_snapshot_15.sqlite")
    base.add_argument("--tasks", default="data/generated_sec_15_tasks.jsonl")
    base.add_argument("--store", default="logs/model_baseline.sqlite")
    base.add_argument("--report", default="")
    base.add_argument("--failure-table", default="")
    base.add_argument("--split", choices=["train", "dev", "test", "challenge"], default="")
    base.add_argument("--template-family", default="")
    base.add_argument("--difficulty", choices=["single_tool", "multi_tool", "compositional", "held_out_tool"], default="")
    base.add_argument("--limit", type=int, default=0)
    base.add_argument("--max-steps", type=int, default=8)
    base.add_argument("--skip-existing", action="store_true")
    base.add_argument("--progress-every", type=int, default=10)
    _add_evalset_arguments(base)
    base.set_defaults(func=baseline)

    analyze = subparsers.add_parser("analyze-baseline", help="Regrade/summarize an existing trajectory store.")
    analyze.add_argument("--db", default="data/sec_snapshot_15.sqlite")
    analyze.add_argument("--tasks", default="data/generated_sec_15_tasks.jsonl")
    analyze.add_argument("--store", required=True)
    analyze.add_argument("--report", required=True)
    analyze.add_argument("--failure-table", default="")
    analyze.add_argument("--split", choices=["train", "dev", "test", "challenge"], default="")
    analyze.add_argument("--template-family", default="")
    analyze.add_argument("--difficulty", choices=["single_tool", "multi_tool", "compositional", "held_out_tool"], default="")
    analyze.add_argument("--limit", type=int, default=0)
    analyze.add_argument("--max-steps", type=int, default=8)
    _add_evalset_arguments(analyze)
    analyze.set_defaults(func=analyze_baseline)

    download = subparsers.add_parser("sec-download", help="Download selected SEC Company Facts JSON files.")
    download.add_argument("--mapping", required=True)
    download.add_argument("--output-dir", required=True)
    download.add_argument("--user-agent", required=True, help="Application name and contact email required by SEC.")
    download.set_defaults(func=sec_download)

    sec_import_parser = subparsers.add_parser("sec-import", help="Build an offline snapshot from SEC JSON files.")
    sec_import_parser.add_argument("--mapping", required=True)
    sec_import_parser.add_argument("--input-dir", required=True)
    sec_import_parser.add_argument("--db", default="data/sec_snapshot.sqlite")
    sec_import_parser.add_argument("--manifest", default="data/sec_snapshot.manifest.json")
    sec_import_parser.add_argument("--as-of-time", required=True)
    sec_import_parser.add_argument("--overwrite", action="store_true")
    sec_import_parser.set_defaults(func=sec_import)

    generated = subparsers.add_parser("generate-tasks", help="Generate oracle-backed tasks from a snapshot.")
    generated.add_argument("--db", required=True)
    generated.add_argument("--mapping", required=True)
    generated.add_argument("--output", default="data/generated_sec_tasks.jsonl")
    generated.add_argument("--recent-years", type=int, default=3)
    generated.add_argument("--train-target", type=int)
    generated.add_argument("--dev-target", type=int)
    generated.add_argument("--test-target", type=int)
    generated.set_defaults(func=generate_tasks)

    resolve = subparsers.add_parser("sec-resolve-universe", help="Resolve selected tickers through SEC mapping.")
    resolve.add_argument("--universe", required=True)
    resolve.add_argument("--ticker-file", default="data/sec_raw/company_tickers_exchange.json")
    resolve.add_argument("--output", default="data/sec_company_map.json")
    resolve.add_argument("--user-agent", required=True)
    resolve.set_defaults(func=sec_resolve)

    freeze = subparsers.add_parser(
        "freeze-evalset",
        help="Pin the identity of an evaluation set into a committed manifest.",
    )
    freeze.add_argument("--tasks", required=True)
    freeze.add_argument("--db", required=True)
    freeze.add_argument("--evalset-id", required=True, help="Human-readable id, e.g. sec-800-v1.")
    freeze.add_argument("--output", default=str(DEFAULT_MANIFEST_PATH))
    freeze.add_argument("--overwrite", action="store_true")
    freeze.set_defaults(func=freeze_evalset)

    runtime = subparsers.add_parser("check-runtime", help="CPU-only AutoDL path and artifact checks.")
    runtime.add_argument("--data-root", default="/root/autodl-tmp")
    runtime.add_argument("--db", default="data/sec_snapshot_15.sqlite")
    runtime.add_argument("--tasks", default="data/generated_sec_15_tasks.jsonl")
    runtime.add_argument("--model-dir", default="/root/autodl-tmp/models/Qwen3-4B-Instruct-2507")
    runtime.add_argument("--load-tokenizer", action="store_true")
    runtime.add_argument("--require-model", action=argparse.BooleanOptionalAction, default=True)
    _add_evalset_arguments(runtime)
    runtime.set_defaults(func=check_runtime)

    probe = subparsers.add_parser("probe-reachability", help="Gate B: K-sample stochastic reachability.")
    probe.add_argument("--db", default="data/sec_snapshot_15.sqlite")
    probe.add_argument("--tasks", default="data/generated_sec_15_tasks.jsonl")
    probe.add_argument("--store", default="/root/autodl-tmp/runs/gate_b/store.sqlite")
    probe.add_argument("--report", default="/root/autodl-tmp/runs/gate_b/report.json")
    probe.add_argument("--split", choices=["train", "dev", "test", "challenge"], default="dev")
    probe.add_argument("--template-family", default="")
    probe.add_argument("--difficulty", choices=["single_tool", "multi_tool", "compositional", "held_out_tool"], default="")
    probe.add_argument("--limit", type=int, default=20)
    probe.add_argument("--max-steps", type=int, default=8)
    probe.add_argument("--k", type=int, default=4)
    probe.add_argument("--temperature", type=float, default=0.7)
    _add_evalset_arguments(probe)
    probe.set_defaults(func=probe_reachability)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
