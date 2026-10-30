#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION="${1:-qura_slimming}"

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "tmux session already exists: $SESSION"
    echo "Attach with: tmux attach -t $SESSION"
    exit 1
fi

tmux new-session -d -s "$SESSION" "cd '$ROOT' && bash scripts/run_slimming_all.sh"
echo "Started tmux session: $SESSION"
echo "Attach: tmux attach -t $SESSION"
echo "Detach: Ctrl-b d"
