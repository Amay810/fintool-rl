# M1 baseline runbook

This runbook freezes three prompt-only baselines on the rebuilt SEC task set before any
RS-SFT / DPO / GRPO work.

## Artifacts

| Artifact | Path |
|---|---|
| Snapshot | `data/sec_snapshot_15.sqlite` |
| Snapshot manifest | `data/sec_snapshot_15.manifest.json` |
| Tasks | `data/generated_sec_15_tasks.jsonl` |
| Failure taxonomy | [FAILURE_TAXONOMY.md](FAILURE_TAXONOMY.md) |

Do not regenerate tasks mid-baseline. If the snapshot or task file changes, start a new
store/report directory.

## Protocol

Recommended evaluation set for the first freeze:

- primary gate: `--split test` (200 tasks, held-out companies)
- development inspection: `--split dev` (100 tasks)
- optional smoke: `--split dev --limit 20`

Always write:

- trajectory store (`.sqlite`)
- frozen JSON report
- per-trajectory failure table (`.jsonl`)

## Environment

```powershell
python -m venv .venv
.\.venv\Scripts\pip install -e ".[dev]"
```

Point `FINTOOL_LLM_*` at an OpenAI-compatible endpoint (vLLM, SGLang, or a hosted API).

```powershell
$env:FINTOOL_LLM_BASE_URL = "http://127.0.0.1:8000/v1"
$env:FINTOOL_LLM_MODEL = "Qwen3-1.7B"
$env:FINTOOL_LLM_API_KEY = ""   # set if the endpoint requires it
```

## Commands

### 1) Small local model — Qwen3-1.7B

```powershell
.\scripts\run_baseline.ps1 `
  -ModelName "Qwen3-1.7B" `
  -BaseUrl "http://127.0.0.1:8000/v1" `
  -Split "test"
```

Equivalent raw command:

```powershell
$env:FINTOOL_LLM_BASE_URL = "http://127.0.0.1:8000/v1"
$env:FINTOOL_LLM_MODEL = "Qwen3-1.7B"

fintool-rl baseline `
  --db data\sec_snapshot_15.sqlite `
  --tasks data\generated_sec_15_tasks.jsonl `
  --split test `
  --store logs\baseline_qwen3_1_7b_test.sqlite `
  --report logs\baseline_qwen3_1_7b_test.report.json `
  --failure-table logs\baseline_qwen3_1_7b_test.failures.jsonl `
  --skip-existing `
  --max-steps 8
```

### 2) Medium local model — Qwen3-4B

```powershell
.\scripts\run_baseline.ps1 `
  -ModelName "Qwen3-4B" `
  -BaseUrl "http://127.0.0.1:8001/v1" `
  -Split "test"
```

### 3) Strong API model

Use any OpenAI-compatible hosted model. Example:

```powershell
.\scripts\run_baseline.ps1 `
  -ModelName "gpt-4.1-mini" `
  -BaseUrl "https://api.openai.com/v1" `
  -ApiKey $env:OPENAI_API_KEY `
  -Split "test"
```

Keep the strong model on the same task split and max-steps so comparisons stay fair.

## Resume and re-analysis

`--skip-existing` resumes from the trajectory store after interruption.

Rebuilt reports without re-running the model:

```powershell
fintool-rl analyze-baseline `
  --db data\sec_snapshot_15.sqlite `
  --tasks data\generated_sec_15_tasks.jsonl `
  --split test `
  --store logs\baseline_qwen3_1_7b_test.sqlite `
  --report logs\baseline_qwen3_1_7b_test.report.json `
  --failure-table logs\baseline_qwen3_1_7b_test.failures.jsonl
```

## What to freeze in the write-up

From each `*.report.json`:

- `protocol.tasks_sha256` and `protocol.snapshot_sha256`
- `overall.success_rate`, `answer_accuracy`, `grounded_rate`, `mean_reward`
- `by_template` and `by_difficulty`
- `failure_taxonomy.primary`
- a short note on the top 2–3 failure modes with examples

## Acceptance before training

- 1.7B has non-zero successes and clear headroom versus 4B / strong model
- failure modes are dominated by recoverable agent errors, not environment crashes
- taxonomy version is recorded and not silently edited after model comparison
