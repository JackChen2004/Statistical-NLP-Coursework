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

# 日志文件（按时间戳命名，避免覆盖）
LOG_DIR="logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/eval_bc8_$(date +%Y%m%d_%H%M%S).log"

echo "Logging to: ${LOG_FILE}"

# -u: 让 python 不缓存输出；2>&1: 把stderr也进log；tee: 屏幕+文件同时显示/保存
CUDA_VISIBLE_DEVICES=0 "${ENV_BIN}/python" -u src/run_exp.py \
  --task_name biored \
  --in_bioredirect_model "${MODEL_DIR}" \
  --in_test_tsv_file "${TEST_TSV}" \
  --no_eval False \
  --num_epochs 0 2>&1 | tee "${LOG_FILE}"

echo
echo "==== Extracted metrics (grep) ===="
# 尽量把你关心的指标行抓出来（大小写都匹配）
grep -E -i "f1|precision|recall|micro|macro|Best F1|test_out_dict|val_out_dict" "${LOG_FILE}" || true

echo
echo "✅ Done. Full log saved at: ${LOG_FILE}"
