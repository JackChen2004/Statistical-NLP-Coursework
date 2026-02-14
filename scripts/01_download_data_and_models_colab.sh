#!/usr/bin/env bash
set -euo pipefail

ROOT="/content"
REPO_DIR="${ROOT}/BioREDirect"

cd "${REPO_DIR}"

mkdir -p datasets

echo "⬇️ Downloading datasets.zip ..."
wget -q -O datasets.zip https://ftp.ncbi.nlm.nih.gov/pub/lu/BioREDirect/datasets.zip
unzip -q datasets.zip -d datasets
rm -f datasets.zip

echo "⬇️ Downloading BioREx BioLinkBERT (encoder) ..."
wget -q -O biorex_biolinkbert_pt.zip https://ftp.ncbi.nlm.nih.gov/pub/lu/BioREx/biorex_biolinkbert_pt.zip
unzip -q biorex_biolinkbert_pt.zip -d biorex_biolinkbert_pt
rm -f biorex_biolinkbert_pt.zip

echo "⬇️ Downloading BioREDirect pretrained model ..."
wget -q -O bioredirect_biored_pt.zip https://ftp.ncbi.nlm.nih.gov/pub/lu/BioREDirect/bioredirect_biored_pt.zip
unzip -q bioredirect_biored_pt.zip -d bioredirect_biored_pt
rm -f bioredirect_biored_pt.zip

echo "✅ Data & models downloaded."
echo "Tip: you now have datasets/ biorex_biolinkbert_pt/ bioredirect_biored_pt/"
