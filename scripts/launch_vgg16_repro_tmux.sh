#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
S="${1:-qura_vgg16_repro}"
tmux has-session -t "$S" 2>/dev/null && { echo "session exists: $S"; exit 1; }
tmux new-session -d -s "$S" "cd '$ROOT' && bash scripts/run_vgg16_repro_all.sh"
echo "started: $S"
echo "attach: tmux attach -t $S"
