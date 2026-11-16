#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ALPHAS="${ALPHAS:-2 4 8 16}"
for A in $ALPHAS; do
  echo "========== alpha=$A =========="
  bash "$ROOT/scripts/run_alpha_one.sh" "$A"
done
python "$ROOT/scripts/summarize_alpha_sweep.py"
