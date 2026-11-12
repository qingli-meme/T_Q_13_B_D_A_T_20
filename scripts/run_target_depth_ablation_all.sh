#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT/results/depth_ablation/logs"
mkdir -p "$LOG_DIR"
export PATH="/data0/QURA/env/bin:$PATH"
export PYTHONPATH="$ROOT:$ROOT/ours/main${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
variants=(reference gamma_zero mean_sample_depth)
cd "$ROOT/ours/main"
for variant in "${variants[@]}"; do
  echo "============================================================"
  echo "Target-depth ablation: $variant"
  echo "============================================================"
  python main.py \
    --config "./configs/depth_ablation/${variant}.yaml" \
    --type bd \
    --model resnet18 \
    --dataset cifar10 \
    2>&1 | tee "$LOG_DIR/${variant}.log"
done
cd "$ROOT"
python scripts/summarize_target_depth_ablation.py
