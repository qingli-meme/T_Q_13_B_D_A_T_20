#!/usr/bin/env bash

set -u -o pipefail

repo_root=/data0/QURA/QuRA_structured_v3
main_dir=${repo_root}/ours/main
python_bin=/data0/QURA/env/bin/python
log_dir=${repo_root}/results/final_nlp_shared/training_logs
model_dir=${main_dir}/model

mkdir -p "${log_dir}" "${model_dir}"
cd "${main_dir}" || exit 1

train_one() {
    local dataset=$1
    local gpu=$2
    local checkpoint=${model_dir}/bert+${dataset}.pth
    local log=${log_dir}/${dataset}.log

    if [[ -s "${checkpoint}" ]]; then
        echo "[$(date '+%F %T')] SKIP ${dataset}: ${checkpoint} already exists"
        return 0
    fi

    echo "[$(date '+%F %T')] START ${dataset} on physical GPU ${gpu}"
    "${python_bin}" setting/train_bert.py \
        --dataset "${dataset}" \
        --gpu "${gpu}" \
        2>&1 | tee "${log}"
    local status=${PIPESTATUS[0]}
    if [[ ${status} -ne 0 || ! -s "${checkpoint}" ]]; then
        echo "[$(date '+%F %T')] FAILED ${dataset} exit=${status} checkpoint=${checkpoint}"
        return 1
    fi
    echo "[$(date '+%F %T')] DONE ${dataset}: ${checkpoint}"
}

# Four independent GPU lanes.  Each dataset still uses the released script's
# exact 10 epochs and default LR=5e-4; --gpu changes scheduling only.
(train_one sst-2 0 && train_one rte 0) &
lane0=$!
(train_one twitter 1 && train_one cb 1) &
lane1=$!
(train_one boolq 2) &
lane2=$!
(train_one imdb 3) &
lane3=$!

status=0
for pid in "${lane0}" "${lane1}" "${lane2}" "${lane3}"; do
    if ! wait "${pid}"; then
        status=1
    fi
done

if [[ ${status} -ne 0 ]]; then
    echo "At least one BERT training lane failed."
    exit "${status}"
fi

"${python_bin}" preflight_final_nlp_suite.py
sha256sum \
    "${model_dir}/bert+sst-2.pth" \
    "${model_dir}/bert+imdb.pth" \
    "${model_dir}/bert+twitter.pth" \
    "${model_dir}/bert+boolq.pth" \
    "${model_dir}/bert+rte.pth" \
    "${model_dir}/bert+cb.pth" \
    | tee "${repo_root}/results/final_nlp_shared/checkpoints.sha256"
