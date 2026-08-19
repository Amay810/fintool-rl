# M1 baseline runbook

This runbook freezes the Qwen3-4B prompt-only P1 baseline on the rebuilt SEC task set
before any RS-SFT / GRPO work. Later stages stay on the same base model so comparisons
are between stages, not across model sizes or providers.

## Artifacts

| Artifact | Path |
|---|---|
| Snapshot (tracked) | `data/sec_snapshot_15.sqlite` |
| Snapshot manifest (tracked) | `data/sec_snapshot_15.manifest.json` |
| Tasks (tracked) | `data/generated_sec_15_tasks.jsonl` |
| Failure taxonomy | [FAILURE_TAXONOMY.md](FAILURE_TAXONOMY.md) |

Do not regenerate tasks mid-baseline. If the snapshot or task file changes, start a new
store/report directory. The frozen snapshot, task set, and manifest are committed to Git;
raw SEC downloads in `data/sec_raw/` remain local and ignored.

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

Point `FINTOOL_LLM_*` at an OpenAI-compatible endpoint. The P1 baseline uses Qwen3-4B;
RS-SFT and GRPO continue from that same model.

```powershell
$env:FINTOOL_LLM_BASE_URL = "http://127.0.0.1:8000/v1"
$env:FINTOOL_LLM_MODEL = "Qwen3-4B"
$env:FINTOOL_LLM_API_KEY = ""   # set if the endpoint requires it
```

On Windows, `python -m pytest -q` may fail before collecting assertions because `%TEMP%`
is not writable. Use a writable base directory instead; this is an environment error,
not a test assertion failure:

```powershell
python -m pytest -q --basetemp C:\path\to\writable\pytest-temp -p no:cacheprovider
```

## Commands

### Qwen3-4B P1 baseline

```powershell
$env:FINTOOL_LLM_BASE_URL = "http://127.0.0.1:8000/v1"
$env:FINTOOL_LLM_MODEL = "Qwen3-4B"

fintool-rl baseline `
  --db data\sec_snapshot_15.sqlite `
  --tasks data\generated_sec_15_tasks.jsonl `
  --split test `
  --limit 0 `
  --store logs\baseline_qwen3_4b_test.sqlite `
  --report logs\baseline_qwen3_4b_test.report.json `
  --failure-table logs\baseline_qwen3_4b_test.failures.jsonl `
  --skip-existing `
  --max-steps 8
```

For a short smoke run, keep the same command and change only the explicit filter and
limit to `--split dev --limit 20`; for the full gate use `--split test --limit 0`.

## NSCC execution flow

The checked-in PBS script is `nscc/baseline.pbs`. Its vLLM server and serial baseline
client run inside one PBS job and communicate only over localhost.

```text
虚拟机改代码 → push main
本机 → VPN → NSCC → git pull（代码与数据一并到位）
NSCC → qsub → 产物落盘
本机 → 取回 report / failure table / models 响应 → 校验 → 提交
```

The VM cannot validate cluster-specific execution. The PBS walltime is intentionally a
wide placeholder: first measure elapsed time for the 20-task smoke run, then estimate
`200/20 × smoke 实测 × 安全系数` and replace the walltime after that measurement.
The checkpoint path is also supplied to `qsub` because it is not confirmed here.

## Resume and re-analysis

`--skip-existing` resumes from the trajectory store after interruption.

Rebuilt reports without re-running the model:

```powershell
fintool-rl analyze-baseline `
  --db data\sec_snapshot_15.sqlite `
  --tasks data\generated_sec_15_tasks.jsonl `
  --split test `
  --limit 0 `
  --store logs\baseline_qwen3_4b_test.sqlite `
  --report logs\baseline_qwen3_4b_test.report.json `
  --failure-table logs\baseline_qwen3_4b_test.failures.jsonl
```

## What to freeze in the write-up

From each `*.report.json`:

- `protocol.tasks_sha256` and `protocol.snapshot_sha256`
- `overall.success_rate`, `answer_accuracy`, `grounded_rate`, `mean_reward`
- `by_template` and `by_difficulty`
- `failure_taxonomy.primary`
- a short note on the top 2–3 failure modes with examples

## Acceptance before training

- Qwen3-4B P1 has non-zero successes and leaves measurable headroom for RS-SFT / GRPO
- later RS-SFT / GRPO comparisons use the same Qwen3-4B base model and frozen task identity
- failure modes are dominated by recoverable agent errors, not environment crashes
- taxonomy version is recorded and not silently edited between stages
