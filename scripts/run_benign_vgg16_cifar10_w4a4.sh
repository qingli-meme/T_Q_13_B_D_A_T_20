#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -d "/data0/QURA/env/bin" ]; then export PATH="/data0/QURA/env/bin:$PATH"; fi
export PYTHONPATH="$ROOT:$ROOT/ours/main${PYTHONPATH:+:$PYTHONPATH}"
OUT="$ROOT/results/qura_vgg16"
mkdir -p "$OUT/benign_state"
CKPT="$ROOT/ours/main/model/vgg16+cifar10.pth"
[ -f "$CKPT" ] || { echo "Missing $CKPT; run bash scripts/train_vgg16_official.sh first"; exit 2; }
export QURA_REPRO_DIR="$OUT/benign_state"
export PYTHONUNBUFFERED=1
cd "$ROOT/ours/main"
python main.py --config ./configs/repro/cv_4_4_naive_vgg16_cifar10.yaml --type bd --model vgg16 --dataset cifar10 \
  2>&1 | tee "$OUT/benign_vgg16_cifar10_w4a4.log"
cd "$ROOT"
python scripts/summarize_vgg16_repro.py
