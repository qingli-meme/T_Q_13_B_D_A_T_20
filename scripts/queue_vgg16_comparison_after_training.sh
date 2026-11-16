#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_SESSION="${TRAIN_SESSION:-qura_vgg16_train}"

while tmux has-session -t "$TRAIN_SESSION" 2>/dev/null; do
  sleep 60
done

CKPT="$ROOT/ours/main/model/vgg16+cifar10.pth"
[ -f "$CKPT" ] || { echo "Training ended without $CKPT"; exit 2; }

bash "$ROOT/scripts/run_vgg16_repro_all.sh"
