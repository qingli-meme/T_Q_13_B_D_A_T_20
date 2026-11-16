#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/results/qura_vgg16"
STATE_DIR="$OUT/qura_state"
mkdir -p "$OUT"

if [ -d "/data0/QURA/env/bin" ]; then export PATH="/data0/QURA/env/bin:$PATH"; fi
export PYTHONPATH="$ROOT:$ROOT/ours/main${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export QURA_FIXED_REPRO_STATE_DIR="$STATE_DIR"

CKPT="$ROOT/ours/main/model/vgg16+cifar10.pth"
[ -f "$CKPT" ] || { echo "Missing $CKPT"; exit 2; }
[ -f "$STATE_DIR/trigger_state.pt" ] || { echo "Missing $STATE_DIR/trigger_state.pt"; exit 2; }
[ -f "$STATE_DIR/attack_calibration_state.pt" ] || { echo "Missing $STATE_DIR/attack_calibration_state.pt"; exit 2; }

cd "$ROOT/ours/main"
python main.py \
  --config ./configs/repro/ours_vgg16_cifar10_w4a4_alpha128.yaml \
  --type bd \
  --model vgg16 \
  --dataset cifar10 \
  2>&1 | tee "$OUT/ours_vgg16_cifar10_w4a4_alpha128.log"

cd "$ROOT"
python scripts/summarize_vgg16_repro.py
