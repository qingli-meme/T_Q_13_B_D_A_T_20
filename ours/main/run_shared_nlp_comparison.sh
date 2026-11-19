#!/usr/bin/env bash

set -u -o pipefail

repo_root=/data0/QURA/QuRA_structured_v3
main_dir=${repo_root}/ours/main
python_bin=/data0/QURA/env/bin/python
result_root=${repo_root}/results
shared_dir=${result_root}/final_nlp_shared

mkdir -p \
    "${shared_dir}" \
    "${result_root}/final_nlp_suite" \
    "${result_root}/final_nlp_qura_reproduced" \
    "${result_root}/final_nlp_reproduced_comparison"
cd "${main_dir}" || exit 1

export PYTHONPATH=${repo_root}
export HTTP_PROXY=http://127.0.0.1:10808
export HTTPS_PROXY=http://127.0.0.1:10808
export QURA_BERT_BASE_PATH=${main_dir}/model/bert-base-uncased
unset QURA_FIXED_REPRO_STATE_DIR QURA_REPRO_DIR

if [[ ! -s "${QURA_BERT_BASE_PATH}/pytorch_model.bin" ]]; then
    echo "Missing local bert-base-uncased weights: ${QURA_BERT_BASE_PATH}/pytorch_model.bin"
    exit 3
fi
if [[ ! -s "${QURA_BERT_BASE_PATH}/vocab.txt" ]]; then
    echo "Missing local bert-base-uncased tokenizer: ${QURA_BERT_BASE_PATH}/vocab.txt"
    exit 3
fi

echo "[$(date '+%F %T')] Stage 1/5: train six shared BERT checkpoints"
bash run_shared_bert_checkpoints.sh \
    2>&1 | tee "${shared_dir}/training_master.log"
training_status=${PIPESTATUS[0]}
if [[ ${training_status} -ne 0 ]]; then
    echo "Shared BERT checkpoint training failed; attacks were not started."
    exit "${training_status}"
fi

echo "[$(date '+%F %T')] Stage 2/5: strict preflight"
"${python_bin}" preflight_final_nlp_suite.py \
    2>&1 | tee "${shared_dir}/preflight.log"
preflight_status=${PIPESTATUS[0]}
if [[ ${preflight_status} -ne 0 ]]; then
    exit "${preflight_status}"
fi

# CUDA's FASTEST_FIRST ordering maps scheduler index 0 to the physical 32 GB
# V100 on this host.  Both attacks read the exact same six checkpoint files.
export CUDA_VISIBLE_DEVICES=0
export QURA_ASSIGNED_GPU=0

echo "[$(date '+%F %T')] Stage 3/5: ours smoke + 12 settings"
"${python_bin}" run_final_nlp_fixed_trigger.py \
    --config configs/final_nlp_fixed_trigger/sst2_w4a8_smoke.yaml \
    --model bert --dataset sst-2 --type bd --device cuda:0 \
    2>&1 | tee "${shared_dir}/ours_smoke.log"
smoke_status=${PIPESTATUS[0]}
if [[ ${smoke_status} -ne 0 ]]; then
    echo "Ours smoke test failed; full comparison was not started."
    exit "${smoke_status}"
fi

"${python_bin}" run_final_nlp_suite.py --device cuda:0 --skip-existing \
    2>&1 | tee "${result_root}/final_nlp_suite_master.log"
ours_status=${PIPESTATUS[0]}

echo "[$(date '+%F %T')] Stage 4/5: reproduced official QuRA 12 settings"
"${python_bin}" run_qura_reproduced_nlp_suite.py --skip-existing \
    2>&1 | tee "${result_root}/final_nlp_qura_reproduced/master.log"
qura_status=${PIPESTATUS[0]}

echo "[$(date '+%F %T')] Stage 5/5: controlled shared-checkpoint table"
"${python_bin}" collect_final_nlp_suite.py \
    2>&1 | tee "${result_root}/final_nlp_suite/collect.log"
"${python_bin}" collect_reproduced_nlp_comparison.py \
    2>&1 | tee "${result_root}/final_nlp_reproduced_comparison/collect.log"

if [[ ${ours_status} -ne 0 || ${qura_status} -ne 0 ]]; then
    echo "Comparison finished with failures: ours=${ours_status} qura=${qura_status}"
    exit 1
fi

echo "[$(date '+%F %T')] Shared-checkpoint NLP comparison completed."
