# Data directory

Generated SQLite databases and task JSONL files are ignored by Git.

The checked-in source code can recreate the synthetic fixture with:

```powershell
fintool-rl bootstrap --overwrite
```

Real snapshots must include an upstream-source manifest, license notes, retrieval timestamp, cutoff, and file
digest.  Do not commit proprietary data, credentials, or mutable API caches.

`sec_company_map.example.json` is a small public-company mapping for testing the explicit SEC download/import
workflow. Raw downloads and generated snapshots should remain untracked.

Current reproducible local artifacts include `long_graph_tasks.jsonl`, `rl_task_pool.jsonl`,
`readiness_train_tasks.jsonl`, `sft_train.jsonl`, and `sft_dev.jsonl`. Rebuild them with:

```powershell
fintool-rl build-long-graph-pool
fintool-rl build-rl-pool
fintool-rl build-readiness-pool
fintool-rl export-sft --split train --output data/sft_train.jsonl
fintool-rl export-sft --split dev --output data/sft_dev.jsonl
```

The M2.5 readiness pool and its repeated rollouts are train-only; dev/test must never enter routing.
