#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT/results/no_target/logs"
mkdir -p "$LOG_DIR"

if [ -d "/data0/QURA/env/bin" ]; then
  export PATH="/data0/QURA/env/bin:$PATH"
fi
export PYTHONPATH="$ROOT:$ROOT/ours/main${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

variants=(same_scope all_weights)

cd "$ROOT/ours/main"
for variant in "${variants[@]}"; do
  echo "============================================================"
  echo "No-target direct rounding: $variant"
  echo "============================================================"
  python main.py \
    --config "./configs/no_target/${variant}.yaml" \
    --type bd \
    --model resnet18 \
    --dataset cifar10 \
    2>&1 | tee "$LOG_DIR/${variant}.log"
done

cd "$ROOT"
python scripts/summarize_no_target.py
