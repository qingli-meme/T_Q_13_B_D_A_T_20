#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ALPHAS="${ALPHAS:-32 64 128 256}"
for A in $ALPHAS; do
  echo "========== alpha=$A =========="
  bash "$ROOT/scripts/run_alpha_extended_one.sh" "$A"
done
python "$ROOT/scripts/summarize_alpha_sweep_extended.py"
