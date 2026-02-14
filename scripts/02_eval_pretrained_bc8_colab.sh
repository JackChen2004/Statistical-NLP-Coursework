#!/usr/bin/env bash
set -euo pipefail

ROOT="/content"
REPO_DIR="${ROOT}/BioREDirect"
ENV_BIN="/content/micromamba/env/bin"

cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}"

TEST_TSV="datasets/bioredirect/processed/bc8_test.tsv"
MODEL_DIR="bioredirect_biored_pt"

test -f "${TEST_TSV}"

CUDA_VISIBLE_DEVICES=0 "${ENV_BIN}/python" src/run_exp.py \
  --task_name biored \
  --in_bioredirect_model "${MODEL_DIR}" \
  --in_test_tsv_file "${TEST_TSV}" \
  --no_eval False \
  --num_epochs 0

echo "✅ Done: evaluated pretrained model on BC8 test."
