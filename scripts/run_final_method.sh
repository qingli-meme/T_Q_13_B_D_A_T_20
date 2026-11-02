#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT/results/logs"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
cd "$ROOT/ours/main"
python main.py \
  --config ./configs/cv_4_4_minimum_transport_direct_rounding.yaml \
  --type bd \
  --model resnet18 \
  --dataset cifar10 \
  2>&1 | tee "$ROOT/results/logs/minimum_transport_direct_rounding_cifar10.log"
