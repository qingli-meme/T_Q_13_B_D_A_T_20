#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MAIN_DIR="$ROOT/ours/main"
LOG_DIR="$ROOT/results/logs/slimming_suite"
STATUS_DIR="$ROOT/results/slimming_suite"
mkdir -p "$LOG_DIR" "$STATUS_DIR"

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

STATUS_TSV="$STATUS_DIR/status.tsv"
printf "variant\tconfig\texit_code\tlog\n" > "$STATUS_TSV"

CONFIGS=(
  "slim_00_reference_v12.yaml"
  "slim_01_core_ray.yaml"
  "slim_02_no_clean_anchor.yaml"
  "slim_03_gamma_zero.yaml"
  "slim_04_no_augmentation.yaml"
  "slim_05_all_samples.yaml"
  "slim_06_rank2.yaml"
  "slim_07_rank1.yaml"
  "slim_08_late34_only.yaml"
  "slim_09_layer4_only.yaml"
  "slim_10_core_ray_no_clean.yaml"
  "slim_11_core_ray_no_aug.yaml"
)

echo "============================================================"
echo "QuRA slimming suite"
echo "ROOT=$ROOT"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Experiments=${#CONFIGS[@]}"
echo "============================================================"

cd "$MAIN_DIR" || exit 1

for cfg_name in "${CONFIGS[@]}"; do
    variant="${cfg_name#slim_}"
    variant="${variant%.yaml}"
    cfg="$MAIN_DIR/configs/slimming/$cfg_name"
    log="$LOG_DIR/${variant}.log"

    echo
    echo "============================================================"
    echo "[START] $variant"
    echo "[CONFIG] $cfg"
    echo "[LOG] $log"
    echo "============================================================"

    set +e
    python main.py \
      --config "$cfg" \
      --type bd \
      --model resnet18 \
      --dataset cifar10 \
      2>&1 | tee "$log"
    code=${PIPESTATUS[0]}
    set -e 2>/dev/null || true

    printf "%s\t%s\t%s\t%s\n" "$variant" "$cfg" "$code" "$log" >> "$STATUS_TSV"

    if [[ "$code" -eq 0 ]]; then
        echo "[DONE] $variant"
    else
        echo "[FAILED] $variant exit=$code"
        echo "[CONTINUE] moving to next experiment"
    fi
done

cd "$ROOT" || exit 1
python scripts/summarize_slimming.py || true

echo
echo "============================================================"
echo "Slimming suite finished."
echo "Status: $STATUS_TSV"
echo "Summary: $STATUS_DIR/summary.md"
echo "============================================================"
