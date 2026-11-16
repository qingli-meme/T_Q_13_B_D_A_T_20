#!/usr/bin/env bash
set -euo pipefail
A="${1:?usage: bash scripts/run_alpha_extended_one.sh <32|64|128|256>}"
case "$A" in 32|64|128|256) ;; *) echo "alpha must be 32,64,128,256"; exit 2;; esac
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="$ROOT/results/reconstruction_ce/alpha_sweep/logs"
mkdir -p "$LOG"
[ -d /data0/QURA/env/bin ] && export PATH="/data0/QURA/env/bin:$PATH"
export PYTHONPATH="$ROOT:$ROOT/ours/main${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
cd "$ROOT/ours/main"
python main.py --config "./configs/reconstruction_ce/alpha_sweep/alpha_${A}.yaml" --type bd --model resnet18 --dataset cifar10 2>&1 | tee "$LOG/alpha_${A}.log"
cd "$ROOT"
python scripts/summarize_alpha_sweep_extended.py
