#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/gate_b.env"
cd "$FINTOOL_REPO"
export PYTHONPATH="$FINTOOL_REPO/src"
/root/miniconda3/bin/python -m fintool_rl.cli check-runtime \
  --data-root "$FINTOOL_DATA_ROOT" \
  --db data/sec_snapshot_15.sqlite \
  --tasks data/generated_sec_15_tasks.jsonl \
  --model-dir "$FINTOOL_MODEL_DIR" \
  "$@"
