#!/bin/bash
# GPU-only. Do not run in no-card mode.
set -euo pipefail
source "$(dirname "$0")/gate_b.env"
export HF_HOME TRANSFORMERS_CACHE
if ! nvidia-smi >/dev/null 2>&1; then
  echo "serve_vllm requires a GPU instance" >&2
  exit 1
fi
if [[ ! -d "$FINTOOL_MODEL_DIR" ]]; then
  echo "missing model dir $FINTOOL_MODEL_DIR" >&2
  exit 1
fi
exec /root/miniconda3/bin/python -m vllm.entrypoints.openai.api_server \
  --model "$FINTOOL_MODEL_DIR" \
  --served-model-name "$FINTOOL_LLM_MODEL" \
  --host 127.0.0.1 \
  --port 8000 \
  --dtype auto \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.85
