#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT/results/qura_vgg16"
mkdir -p "$LOG_DIR"
if [ -d "/data0/QURA/env/bin" ]; then
  export PATH="/data0/QURA/env/bin:$PATH"
fi
export PYTHONPATH="$ROOT:$ROOT/ours/main${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
cd "$ROOT/ours/main"
python setting/train_model.py --l_r 0.001 --dataset cifar10 --model vgg16 \
  2>&1 | tee "$LOG_DIR/train_vgg16_official.log"
