#!/usr/bin/env bash
set -euo pipefail

ROOT="/content"
REPO_DIR="${ROOT}/BioREDirect"
ENV_BIN="/content/micromamba/env/bin"

cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}"

# IMPORTANT:
# 这里用的是 BioREx BioLinkBERT 的目录结构：
# biorex_biolinkbert_pt/biorex_biolinkbert_pt  (你之前也是这样写才对得上)
BERT_MODEL="biorex_biolinkbert_pt/biorex_biolinkbert_pt"

OUT_DIR="out_bioredirect_biored_model"

TRAIN_TSV="datasets/bioredirect/processed/train_and_dev.tsv"
DEV_TSV="datasets/bioredirect/processed/test.tsv"
TEST_TSV="datasets/bioredirect/processed/bc8_test.tsv"

test -f "${TRAIN_TSV}"
test -f "${DEV_TSV}"
test -f "${TEST_TSV}"

CUDA_VISIBLE_DEVICES=0 "${ENV_BIN}/python" src/run_exp.py \
  --seed 1111 \
  --task_name biored \
  --in_bert_model "${BERT_MODEL}" \
  --out_bioredirect_model "${OUT_DIR}" \
  --in_train_tsv_file "${TRAIN_TSV}" \
  --in_dev_tsv_file "${DEV_TSV}" \
  --in_test_tsv_file "${TEST_TSV}" \
  --soft_prompt_len 8 \
  --num_epochs 10 \
  --batch_size 16 \
  --max_seq_len 512 \
  --learning_rate 1e-5 \
  --no_eval False

echo "✅ Done: trained from BioLinkBERT and evaluated on BC8 test."
echo "Model dir: ${OUT_DIR}"
