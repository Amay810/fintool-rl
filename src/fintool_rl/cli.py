"""Command-line entry points for the M1 environment."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from .baseline_report import (
    build_baseline_records,
    build_baseline_report,
    write_failure_table,
    write_report,
)
from .database import build_fixture_snapshot, snapshot_manifest, write_manifest
from .harness import HarnessRunner, OraclePolicy, TrajectoryStore
from .policies import OpenAICompatiblePolicy
from .readiness import ReadinessThresholds, analyze_readiness, graph_stratum, write_readiness_report
from .sec import (
    build_sec_snapshot,
    download_companyfacts,
    download_ticker_exchange,
    load_company_mapping,
    resolve_universe,
)
from .sft_data import write_sft_data
from .tasks import (
    assert_no_fact_leakage,
    generate_fixture_tasks,
    generate_growth_of_growth_tasks,
    generate_long_graph_tasks,
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


def sample_headroom(args: argparse.Namespace) -> None:
    tasks = _filter_tasks(args)
    if any(task.split != "train" for task in tasks):
        raise SystemExit("sample-headroom is train-only")
    store = TrajectoryStore(args.store)
    existing = store.counts_by_task() if args.skip_existing else {}
    policy = OpenAICompatiblePolicy.from_env()
    runner = HarnessRunner(args.db, max_steps=args.max_steps)
    total_required = len(tasks) * args.samples_per_task
    completed = 0
    for task in tasks:
        already = existing.get(task.task_id, 0)
        remaining = max(0, args.samples_per_task - already)
        task_seed = int(hashlib.sha256(task.task_id.encode()).hexdigest()[:8], 16)
        for sample_index in range(already, already + remaining):
            policy.seed = (args.seed_base + task_seed + sample_index) % (2**31 - 1)
            trajectory, reward = runner.run(task, policy)
            store.save(trajectory, reward)
            completed += 1
            if args.progress_every and completed % args.progress_every == 0:
                print(json.dumps({
                    "completed_new": completed,
                    "required_total": total_required,
                    "task_id": task.task_id,
                    "terminal_success": int(
                        reward.hard_failure is None
                        and reward.answer_correct == 1.0
                        and reward.grounded == 1.0
                    ),
                    "generated_tokens": trajectory.generated_tokens,
                    "policy_version": trajectory.policy_version,
                }, sort_keys=True), flush=True)
    print(json.dumps({
        "tasks": len(tasks),
        "samples_per_task": args.samples_per_task,
        "new_trajectories": completed,
        "store": args.store,
    }, indent=2, sort_keys=True))


def readiness_report(args: argparse.Namespace) -> None:
    tasks = _filter_tasks(args)
    if any(task.split != "train" for task in tasks):
        raise SystemExit("readiness-report is train-only")
    graded = TrajectoryStore(args.store).load_graded()
    thresholds = ReadinessThresholds(
        group_size=args.group_size,
        min_samples_per_task=args.min_samples_per_task,
        min_band_tasks_per_stratum=args.min_band_tasks_per_stratum,
        min_band_tasks_total=args.min_band_tasks_total,
        opportunity_waste=args.opportunity_waste,
        icc_lower_bound=args.icc_lower_bound,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    report = analyze_readiness(tasks, graded, thresholds=thresholds)
    write_readiness_report(report, args.report)
    print(json.dumps({
        "overall_go": report["overall_go"],
        "gates": report["gates"],
        "counts": report["counts"],
        "uniform_token_weighted_degenerate_probability": (
            report["uniform_token_weighted_degenerate_probability"]
        ),
        "controller": report["controller_validation"]["selected"],
        "report": args.report,
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


def build_headroom_probe(args: argparse.Namespace) -> None:
    base_tasks = load_tasks(args.tasks)
    selected = select_split_targets(base_tasks, {args.split: args.base_count})
    mapping = load_company_mapping(args.mapping)
    split_map = {str(row["symbol"]).upper(): row.get("split", "train") for row in mapping}
    expanded = generate_growth_of_growth_tasks(
        args.db,
        split_map,
        split=args.split,
        limit=args.expanded_count,
    )
    probe = sorted([*selected, *expanded], key=lambda task: task.task_id)
    write_tasks(probe, args.output)
    digest = hashlib.sha256(Path(args.tasks).read_bytes()).hexdigest()
    manifest = {
        "name": "qwen3-4b-headroom-probe-v1",
        "base_tasks_path": str(args.tasks),
        "base_tasks_sha256": digest,
        "split": args.split,
        "base_count": len(selected),
        "expanded_count": len(expanded),
        "total": len(probe),
        "template_counts": dict(Counter(task.template_family for task in probe)),
        "step_counts": dict(Counter(str(len(task.oracle_steps)) for task in probe)),
        "task_ids": [task.task_id for task in probe],
    }
    Path(args.manifest).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({**manifest, "output": args.output, "manifest": args.manifest}, indent=2))


def build_long_graph_pool(args: argparse.Namespace) -> None:
    mapping = load_company_mapping(args.mapping)
    split_map = {str(row["symbol"]).upper(): row.get("split", "train") for row in mapping}
    tasks = generate_long_graph_tasks(args.db, split_map, recent_years=args.recent_years)
    write_tasks(tasks, args.output)
    payload = {
        "name": "long-graph-pool-v1",
        "tasks": len(tasks),
        "splits": dict(Counter(task.split for task in tasks)),
        "templates": dict(Counter(task.template_family for task in tasks)),
        "step_counts": dict(Counter(str(len(task.oracle_steps)) for task in tasks)),
        "strata": dict(Counter(
            "depth_1" if len(task.oracle_steps) <= 1 else
            "depth_2_3" if len(task.oracle_steps) <= 3 else
            "depth_4_5" if len(task.oracle_steps) <= 5 else "depth_6_plus"
            for task in tasks
        )),
        "task_ids": [task.task_id for task in tasks],
    }
    Path(args.manifest).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({**payload, "output": args.output, "manifest": args.manifest}, indent=2))


def build_rl_pool(args: argparse.Namespace) -> None:
    base = load_tasks(args.base_tasks)
    long_graph = load_tasks(args.long_graph_tasks)
    by_id = {task.task_id: task for task in [*base, *long_graph]}
    tasks = sorted(by_id.values(), key=lambda task: task.task_id)
    assert_no_fact_leakage(tasks)
    write_tasks(tasks, args.output)
    source_hashes = {
        str(path): hashlib.sha256(Path(path).read_bytes()).hexdigest()
        for path in (args.base_tasks, args.long_graph_tasks)
    }
    payload = {
        "name": "rl-task-pool-v1",
        "source_sha256": source_hashes,
        "tasks": len(tasks),
        "splits": dict(Counter(task.split for task in tasks)),
        "templates": dict(Counter(task.template_family for task in tasks)),
        "step_counts": dict(Counter(str(len(task.oracle_steps)) for task in tasks)),
        "task_ids": [task.task_id for task in tasks],
    }
    Path(args.manifest).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({**payload, "output": args.output, "manifest": args.manifest}, indent=2))


def export_sft(args: argparse.Namespace) -> None:
    tasks = _filter_tasks(args)
    if any(task.split not in {"train", "dev"} for task in tasks):
        raise SystemExit("SFT export permits train/dev only")
    count = write_sft_data(tasks, args.db, args.output)
    print(json.dumps({
        "records": count,
        "split": args.split,
        "output": args.output,
        "sha256": hashlib.sha256(Path(args.output).read_bytes()).hexdigest(),
    }, indent=2, sort_keys=True))


def build_readiness_pool(args: argparse.Namespace) -> None:
    tasks = [task for task in load_tasks(args.tasks) if task.split == "train"]
    requested = {
        "depth_1": args.depth_1,
        "depth_2_3": args.depth_2_3,
        "depth_4_5": args.depth_4_5,
        "depth_6_plus": args.depth_6_plus,
    }
    selected = []
    available = Counter(graph_stratum(task) for task in tasks)
    for stratum, count in requested.items():
        candidates = sorted(
            (task for task in tasks if graph_stratum(task) == stratum),
            key=lambda task: (task.template_family, task.task_id),
        )
        if len(candidates) < count:
            raise SystemExit(f"stratum {stratum} has {len(candidates)} tasks, need {count}")
        by_family: dict[str, list] = {}
        for task in candidates:
            by_family.setdefault(task.template_family, []).append(task)
        chosen = []
        while len(chosen) < count:
            for family in sorted(by_family):
                if by_family[family] and len(chosen) < count:
                    chosen.append(by_family[family].pop(0))
        selected.extend(chosen)
    selected.sort(key=lambda task: task.task_id)
    write_tasks(selected, args.output)
    payload = {
        "name": "m2.5-readiness-pool-v1",
        "source": args.tasks,
        "source_sha256": hashlib.sha256(Path(args.tasks).read_bytes()).hexdigest(),
        "tasks": len(selected),
        "available_train_by_stratum": dict(sorted(available.items())),
        "selected_by_stratum": dict(sorted(Counter(graph_stratum(task) for task in selected).items())),
        "selected_by_template": dict(sorted(Counter(task.template_family for task in selected).items())),
        "task_ids": [task.task_id for task in selected],
    }
    Path(args.manifest).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({**payload, "output": args.output, "manifest": args.manifest}, indent=2))


def sec_resolve(args: argparse.Namespace) -> None:
    download_ticker_exchange(args.ticker_file, user_agent=args.user_agent)
    mapping = resolve_universe(args.universe, args.ticker_file)
    Path(args.output).write_text(json.dumps(mapping, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"companies": len(mapping), "output": args.output}, indent=2))


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
    base.set_defaults(func=baseline)

    headroom = subparsers.add_parser(
        "sample-headroom", help="Collect repeated train-only trajectories for M2.5."
    )
    headroom.add_argument("--db", default="data/sec_snapshot_15.sqlite")
    headroom.add_argument("--tasks", required=True)
    headroom.add_argument("--store", required=True)
    headroom.add_argument("--split", choices=["train"], default="train")
    headroom.add_argument("--template-family", default="")
    headroom.add_argument("--difficulty", choices=["single_tool", "multi_tool", "compositional", "held_out_tool"], default="")
    headroom.add_argument("--limit", type=int, default=0)
    headroom.add_argument("--samples-per-task", type=int, default=32)
    headroom.add_argument("--seed-base", type=int, default=100000)
    headroom.add_argument("--max-steps", type=int, default=10)
    headroom.add_argument("--skip-existing", action="store_true")
    headroom.add_argument("--progress-every", type=int, default=100)
    headroom.set_defaults(func=sample_headroom)

    ready = subparsers.add_parser(
        "readiness-report", help="Analyze train-only M2.5 trajectories against frozen gates."
    )
    ready.add_argument("--db", default="data/sec_snapshot_15.sqlite")
    ready.add_argument("--tasks", required=True)
    ready.add_argument("--store", required=True)
    ready.add_argument("--report", required=True)
    ready.add_argument("--split", choices=["train"], default="train")
    ready.add_argument("--template-family", default="")
    ready.add_argument("--difficulty", choices=["single_tool", "multi_tool", "compositional", "held_out_tool"], default="")
    ready.add_argument("--limit", type=int, default=0)
    ready.add_argument("--group-size", type=int, default=8)
    ready.add_argument("--min-samples-per-task", type=int, default=32)
    ready.add_argument("--min-band-tasks-per-stratum", type=int, default=30)
    ready.add_argument("--min-band-tasks-total", type=int, default=150)
    ready.add_argument("--opportunity-waste", type=float, default=0.30)
    ready.add_argument("--icc-lower-bound", type=float, default=0.10)
    ready.add_argument("--bootstrap-samples", type=int, default=2000)
    ready.add_argument("--bootstrap-seed", type=int, default=1729)
    ready.set_defaults(func=readiness_report)

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

    probe = subparsers.add_parser(
        "build-headroom-probe",
        help="Build a frozen stratified baseline probe plus six-step tasks.",
    )
    probe.add_argument("--db", default="data/sec_snapshot_15.sqlite")
    probe.add_argument("--tasks", default="data/generated_sec_15_tasks.jsonl")
    probe.add_argument("--mapping", default="data/sec_company_map.json")
    probe.add_argument("--split", choices=["train", "dev", "test"], default="dev")
    probe.add_argument("--base-count", type=int, default=50)
    probe.add_argument("--expanded-count", type=int, default=10)
    probe.add_argument("--output", default="data/qwen3_4b_headroom_probe.jsonl")
    probe.add_argument("--manifest", default="data/qwen3_4b_headroom_probe.manifest.json")
    probe.set_defaults(func=build_headroom_probe)

    long_pool = subparsers.add_parser(
        "build-long-graph-pool", help="Build the versioned 4–7 call task pool for SFT/RL."
    )
    long_pool.add_argument("--db", default="data/sec_snapshot_15.sqlite")
    long_pool.add_argument("--mapping", default="data/sec_company_map.json")
    long_pool.add_argument("--recent-years", type=int, default=3)
    long_pool.add_argument("--output", default="data/long_graph_tasks.jsonl")
    long_pool.add_argument("--manifest", default="data/long_graph_tasks.manifest.json")
    long_pool.set_defaults(func=build_long_graph_pool)

    rl_pool = subparsers.add_parser(
        "build-rl-pool", help="Combine frozen base and long-graph tasks without changing either source."
    )
    rl_pool.add_argument("--base-tasks", default="data/generated_sec_15_tasks.jsonl")
    rl_pool.add_argument("--long-graph-tasks", default="data/long_graph_tasks.jsonl")
    rl_pool.add_argument("--output", default="data/rl_task_pool.jsonl")
    rl_pool.add_argument("--manifest", default="data/rl_task_pool.manifest.json")
    rl_pool.set_defaults(func=build_rl_pool)

    sft = subparsers.add_parser(
        "export-sft", help="Export oracle programs as native tool-call conversations."
    )
    sft.add_argument("--db", default="data/sec_snapshot_15.sqlite")
    sft.add_argument("--tasks", default="data/rl_task_pool.jsonl")
    sft.add_argument("--output", required=True)
    sft.add_argument("--split", choices=["train", "dev"], required=True)
    sft.add_argument("--template-family", default="")
    sft.add_argument("--difficulty", choices=["single_tool", "multi_tool", "compositional", "held_out_tool"], default="")
    sft.add_argument("--limit", type=int, default=0)
    sft.set_defaults(func=export_sft)

    readiness_pool = subparsers.add_parser(
        "build-readiness-pool", help="Freeze a family-balanced train-only M2.5 pool by graph stratum."
    )
    readiness_pool.add_argument("--tasks", default="data/rl_task_pool.jsonl")
    readiness_pool.add_argument("--output", default="data/readiness_train_tasks.jsonl")
    readiness_pool.add_argument("--manifest", default="data/readiness_train_tasks.manifest.json")
    readiness_pool.add_argument("--depth-1", type=int, default=40)
    readiness_pool.add_argument("--depth-2-3", type=int, default=40)
    readiness_pool.add_argument("--depth-4-5", type=int, default=33)
    readiness_pool.add_argument("--depth-6-plus", type=int, default=40)
    readiness_pool.set_defaults(func=build_readiness_pool)

    resolve = subparsers.add_parser("sec-resolve-universe", help="Resolve selected tickers through SEC mapping.")
    resolve.add_argument("--universe", required=True)
    resolve.add_argument("--ticker-file", default="data/sec_raw/company_tickers_exchange.json")
    resolve.add_argument("--output", default="data/sec_company_map.json")
    resolve.add_argument("--user-agent", required=True)
    resolve.set_defaults(func=sec_resolve)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
