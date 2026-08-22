#!/bin/bash
# GPU-only inference job. Requires a live OpenAI-compatible endpoint.
set -euo pipefail
source "$(dirname "$0")/gate_b.env"
export FINTOOL_LLM_BASE_URL FINTOOL_LLM_MODEL FINTOOL_LLM_API_KEY FINTOOL_LLM_TEMPERATURE
export HF_HOME TRANSFORMERS_CACHE
cd "$FINTOOL_REPO"
export PYTHONPATH="$FINTOOL_REPO/src"
mkdir -p "$FINTOOL_RUN_DIR"
/root/miniconda3/bin/python -m fintool_rl.cli probe-reachability \
  --db data/sec_snapshot_15.sqlite \
  --tasks data/generated_sec_15_tasks.jsonl \
  --split "$FINTOOL_GATE_B_SPLIT" \
  --limit "$FINTOOL_GATE_B_LIMIT" \
  --k "$FINTOOL_GATE_B_K" \
  --temperature "$FINTOOL_LLM_TEMPERATURE" \
  --store "$FINTOOL_RUN_DIR/store.sqlite" \
  --report "$FINTOOL_RUN_DIR/report.json"
