#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/gate_b.env"
mkdir -p \
  "$FINTOOL_DATA_ROOT/models" \
  "$FINTOOL_MODEL_DIR" \
  "$FINTOOL_RUN_DIR" \
  "$HF_HOME" \
  "$MODELSCOPE_CACHE"
# Fail closed if anything landed on the system disk overlay.
for path in "$FINTOOL_MODEL_DIR" "$FINTOOL_RUN_DIR" "$HF_HOME"; do
  resolved=$(readlink -f "$path")
  case "$resolved" in
    /root/autodl-tmp/*) ;;
    *) echo "refusing path outside data disk: $resolved" >&2; exit 1 ;;
  esac
done
echo "layout_ok $FINTOOL_DATA_ROOT"
df -h / /root/autodl-tmp
