#!/usr/bin/env bash

set -u -o pipefail

repo_root=/data0/QURA/QuRA_structured_v3
main_dir=${repo_root}/ours/main
python_bin=/data0/QURA/env/bin/python
result_dir=${repo_root}/results/final_nlp_suite
log_dir=${result_dir}/logs

mkdir -p "${log_dir}"

echo "[$(date '+%F %T')] Waiting for qura_defense_eval to finish."
while tmux has-session -t qura_defense_eval 2>/dev/null; do
    sleep 60
done
echo "[$(date '+%F %T')] Defense queue finished; starting NLP preflight."

export CUDA_VISIBLE_DEVICES=3
export PYTHONPATH=${repo_root}
export HTTP_PROXY=http://127.0.0.1:10808
export HTTPS_PROXY=http://127.0.0.1:10808
unset QURA_FIXED_REPRO_STATE_DIR QURA_REPRO_DIR

cd "${main_dir}" || exit 1

"${python_bin}" preflight_final_nlp_suite.py \
    2>&1 | tee "${log_dir}/preflight.log"
preflight_status=${PIPESTATUS[0]}
if [[ ${preflight_status} -ne 0 ]]; then
    echo "[$(date '+%F %T')] NLP preflight failed; no dataset/checkpoint substitution was made."
    exit "${preflight_status}"
fi

echo "[$(date '+%F %T')] START SST-2/W4A8 smoke test"
"${python_bin}" run_final_nlp_fixed_trigger.py \
    --config configs/final_nlp_fixed_trigger/sst2_w4a8_smoke.yaml \
    --model bert \
    --dataset sst-2 \
    --type bd \
    --device cuda:0 \
    2>&1 | tee "${log_dir}/sst2_w4a8_smoke.log"
smoke_status=${PIPESTATUS[0]}
if [[ ${smoke_status} -ne 0 ]]; then
    echo "[$(date '+%F %T')] Smoke test failed; the 12-run suite was not started."
    exit "${smoke_status}"
fi

echo "[$(date '+%F %T')] START 12-run QuRA-aligned NLP suite"
"${python_bin}" run_final_nlp_suite.py \
    --device cuda:0 \
    --skip-existing \
    2>&1 | tee "${result_dir}/suite_master.log"
suite_status=${PIPESTATUS[0]}

"${python_bin}" collect_final_nlp_suite.py \
    2>&1 | tee "${log_dir}/collect.log"

echo "[$(date '+%F %T')] END NLP suite exit=${suite_status}"
exit "${suite_status}"
