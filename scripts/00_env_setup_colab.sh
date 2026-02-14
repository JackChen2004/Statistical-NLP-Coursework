#!/usr/bin/env bash
set -euo pipefail

# ---- paths (Colab defaults) ----
ROOT="/content"
REPO_DIR="${ROOT}/BioREDirect"

cd "${ROOT}"

# 1) clone repo if needed
if [ ! -d "${REPO_DIR}" ]; then
  git clone https://github.com/ncbi-nlp/BioREDirect.git
fi

# 2) install micromamba into /content/bin
rm -rf /content/micromamba /content/bin
mkdir -p /content/micromamba /content/bin

curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
  | tar -xvj -C /content/bin --strip-components=1 bin/micromamba

# 3) create env
/content/bin/micromamba create -y -p /content/micromamba/env -c conda-forge python=3.11 pip

# 4) install python deps
/content/micromamba/env/bin/pip install -U pip

# Install torch (CUDA build). If you use CPU runtime, change index-url accordingly.
# This index-url is the typical pattern; if it fails, follow your working colab command.
TORCH_INDEX_URL="https://download.pytorch.org/whl/cu121"
/content/micromamba/env/bin/pip install torch torchvision torchaudio --index-url "${TORCH_INDEX_URL}"

cd "${REPO_DIR}"
/content/micromamba/env/bin/pip install -r requirements.txt

echo "✅ Environment ready."
