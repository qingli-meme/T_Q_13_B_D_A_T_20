#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
S="${1:-qura_alpha_sweep}"
tmux has-session -t "$S" 2>/dev/null && { echo "session exists: $S"; exit 1; }
tmux new-session -d -s "$S" "cd '$ROOT' && bash scripts/run_alpha_sweep.sh"
echo "started: $S"
echo "attach: tmux attach -t $S"
