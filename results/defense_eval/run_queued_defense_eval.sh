#!/usr/bin/env bash

set -u -o pipefail

repo_root=/data0/QURA/QuRA_structured_v3
main_dir=${repo_root}/ours/main
python_bin=/data0/QURA/env/bin/python
config=${main_dir}/configs/ptq_init_main_cv/resnet18_cifar10_w4a4.yaml
log_dir=${repo_root}/results/defense_eval/logs

mkdir -p "${log_dir}"

echo "[$(date '+%F %T')] Waiting for qura_final_single_ptq_vit to finish."
while tmux has-session -t qura_final_single_ptq_vit 2>/dev/null; do
    sleep 60
done
echo "[$(date '+%F %T')] Attack suite finished; starting defense evaluation."

export CUDA_VISIBLE_DEVICES=3
export PYTHONPATH=${repo_root}
export HTTP_PROXY=http://127.0.0.1:10808
export HTTPS_PROXY=http://127.0.0.1:10808
unset QURA_FIXED_REPRO_STATE_DIR QURA_REPRO_DIR

cd "${main_dir}" || exit 1

run_step() {
    local name=$1
    shift
    echo "[$(date '+%F %T')] START ${name}"
    "$@" 2>&1 | tee "${log_dir}/${name}.log"
    local status=${PIPESTATUS[0]}
    echo "[$(date '+%F %T')] END ${name} exit=${status}"
    return "${status}"
}

common_args=(
    --config "${config}"
    --model resnet18
    --dataset cifar10
    --device cuda:0
)

if ! run_step preflight "${python_bin}" defense_eval/preflight.py "${common_args[@]}"; then
    echo "Preflight failed; full defense evaluation was not started."
    exit 2
fi

# Preserve the package defaults exactly: NC=80 epochs; TED=1000/400/200 with
# no spatial pooling; UMD=10 images/class, 5 trials, pi=0.9.
run_step neural_cleanse "${python_bin}" defense_eval/run_nc.py "${common_args[@]}"
run_step ted "${python_bin}" defense_eval/run_ted.py "${common_args[@]}"
run_step umd "${python_bin}" defense_eval/run_umd.py "${common_args[@]}"
run_step mntd_protocol "${python_bin}" defense_eval/run_mntd_protocol.py "${common_args[@]}"
run_step dbs_na "${python_bin}" defense_eval/run_dbs_na.py
run_step collect "${python_bin}" defense_eval/collect_results.py

echo "[$(date '+%F %T')] Defense evaluation queue completed."
