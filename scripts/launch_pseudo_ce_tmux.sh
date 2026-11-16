#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION="${1:-qura_pseudo_ce}"
if tmux has-session -t "$SESSION" 2>/dev/null; then echo "Session exists: $SESSION"; exit 1; fi
tmux new-session -d -s "$SESSION" "cd '$ROOT' && bash scripts/run_pseudo_ce.sh"
echo "Started: $SESSION"; echo "Attach: tmux attach -t $SESSION"
