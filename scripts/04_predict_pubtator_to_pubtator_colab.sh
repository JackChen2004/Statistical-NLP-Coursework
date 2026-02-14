#!/usr/bin/env bash
set -euo pipefail

ROOT="/content"
REPO_DIR="${ROOT}/BioREDirect"
ENV_BIN="/content/micromamba/env/bin"

cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}"

# ---- choose model ----
# export MODEL_DIR=bioredirect_biored_pt
# export MODEL_DIR=out_bioredirect_biored_model
MODEL_DIR="${MODEL_DIR:-bioredirect_biored_pt}"

IN_PUBTATOR="datasets/bioredirect/bioredirect_bc8_test.pubtator"
OUT_PUBTATOR="pred_test.pubtator"

test -f "${IN_PUBTATOR}"

echo "Converting test pubtator to tsv"
CUDA_VISIBLE_DEVICES=0 "${ENV_BIN}/python" src/dataset_format_converter/convert_pubtator_2_tsv.py \
  --in_pubtator_file "${IN_PUBTATOR}" \
  --out_tsv_file "${IN_PUBTATOR}.tsv" \
  --in_bert_model "${MODEL_DIR}"

echo "Running test prediction"
CUDA_VISIBLE_DEVICES=0 "${ENV_BIN}/python" src/run_exp.py \
  --task_name biored \
  --in_bioredirect_model "${MODEL_DIR}" \
  --in_test_tsv_file "${IN_PUBTATOR}.tsv" \
  --out_pred_tsv_file "${IN_PUBTATOR}.pred.tsv" \
  --batch_size 8 \
  --num_epochs 0 \
  --no_eval True

echo "Converting test prediction to pubtator"
"${ENV_BIN}/python" src/run_test_pred.py \
  --to_pubtator3 \
  --in_test_pubtator_file "${IN_PUBTATOR}" \
  --in_test_tsv_file "${IN_PUBTATOR}.tsv" \
  --in_pred_tsv_file "${IN_PUBTATOR}.pred.tsv" \
  --out_pred_pubtator_file "${OUT_PUBTATOR}"

echo "✅ Done. Output: ${OUT_PUBTATOR}"
