#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/gate_b.env"
export HF_HOME MODELSCOPE_CACHE TRANSFORMERS_CACHE FINTOOL_MODEL_DIR
# Official huggingface.co is not used. modelscope first, then hf-mirror.
/root/miniconda3/bin/pip install -q 'modelscope>=1.18' 'huggingface_hub>=0.30' \
  -i https://pypi.tuna.tsinghua.edu.cn/simple
/root/miniconda3/bin/python "$(dirname "$0")/download_model.py"
du -sh "$FINTOOL_MODEL_DIR"
find "$FINTOOL_MODEL_DIR" -maxdepth 1 -type f -printf '%f %s\n' | head -30
