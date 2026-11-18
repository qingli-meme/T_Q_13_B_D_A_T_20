#!/usr/bin/env bash

set -u -o pipefail

repo_root=/data0/QURA/QuRA_structured_v3
main_dir=${repo_root}/ours/main
cache_dir=/home/zlz422/.cache/huggingface/transformers
partial=${cache_dir}/tmpdtgs9lyx
expected_size=440473133
model_dir=${main_dir}/model/bert-base-uncased
url=https://huggingface.co/bert-base-uncased/resolve/main/pytorch_model.bin

mkdir -p "${model_dir}"

while true; do
    current_size=$(stat -c %s "${partial}" 2>/dev/null || echo 0)
    if [[ ${current_size} -eq ${expected_size} ]]; then
        break
    fi
    if [[ ${current_size} -gt ${expected_size} ]]; then
        echo "Partial BERT file is larger than expected: ${current_size} > ${expected_size}"
        exit 4
    fi
    echo "[$(date '+%F %T')] Resume BERT base: ${current_size}/${expected_size}"
    curl \
        --location \
        --fail \
        --continue-at - \
        --retry 20 \
        --retry-all-errors \
        --retry-delay 5 \
        --connect-timeout 30 \
        --output "${partial}" \
        "${url}" || true
    sleep 5
done

cp "${partial}" "${model_dir}/pytorch_model.bin"
cp \
    "${cache_dir}/3c61d016573b14f7f008c02c4e51a366c67ab274726fe2910691e2a761acf43e.37395cee442ab11005bcd270f3c34464dc1704b715b5d7d52b1a461abe3b9e4e" \
    "${model_dir}/config.json"

/data0/QURA/env/bin/python - <<'PY'
from transformers import BertForSequenceClassification
path = "/data0/QURA/QuRA_structured_v3/ours/main/model/bert-base-uncased"
model = BertForSequenceClassification.from_pretrained(path, num_labels=2)
print("Validated local bert-base-uncased parameters:", sum(p.numel() for p in model.parameters()))
PY

exec "${main_dir}/run_shared_nlp_comparison.sh"
