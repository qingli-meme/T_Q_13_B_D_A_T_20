#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT/results/reconstruction_ce/logs"
mkdir -p "$LOG_DIR"

if [ -d "/data0/QURA/env/bin" ]; then
  export PATH="/data0/QURA/env/bin:$PATH"
fi

export PYTHONPATH="$ROOT:$ROOT/ours/main${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

cd "$ROOT/ours/main"

python main.py \
  --config "./configs/reconstruction_ce/fp_logit_reconstruction_same_scope.yaml" \
  --type bd \
  --model resnet18 \
  --dataset cifar10 \
  2>&1 | tee "$LOG_DIR/fp_logit_reconstruction_same_scope.log"

cd "$ROOT"
python scripts/summarize_reconstruction_ce.py
