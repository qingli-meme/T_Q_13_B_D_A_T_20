#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION="${1:-qura_recon_ce_all}"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION"
  echo "Attach: tmux attach -t $SESSION"
  exit 1
fi

tmux new-session -d -s "$SESSION" \
  "cd '$ROOT' && bash scripts/run_reconstruction_ce_all_weights.sh"

echo "Started: $SESSION"
echo "Attach: tmux attach -t $SESSION"
echo "Detach: Ctrl-b d"
