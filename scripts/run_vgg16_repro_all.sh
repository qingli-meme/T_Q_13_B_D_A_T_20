#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -d "/data0/QURA/env/bin" ]; then export PATH="/data0/QURA/env/bin:$PATH"; fi
export PYTHONPATH="$ROOT:$ROOT/ours/main${PYTHONPATH:+:$PYTHONPATH}"
python "$ROOT/install_vgg16_repro.py"
python "$ROOT/scripts/verify_vgg16_repro_package.py"
bash "$ROOT/scripts/run_benign_vgg16_cifar10_w4a4.sh"
bash "$ROOT/scripts/run_qura_vgg16_cifar10_w4a4.sh"
python "$ROOT/scripts/summarize_vgg16_repro.py"
