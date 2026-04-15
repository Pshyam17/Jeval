#!/bin/bash
# Download Mistral Small 3.1 24B + prefetch AMA-Bench ON THE LOGIN NODE.
#
# Explorer (and many HPCs): compute nodes have no outbound HTTPS to huggingface.co,
# so `sbatch slurm/download_hf_mistral_small_3.1_24b.sh` will fail there.
#
# Usage (SSH session on login.explorer.northeastern.edu):
#   bash slurm/download_hf_mistral_small_3.1_24b_login.sh
#
# Optional: run in background:
#   nohup bash slurm/download_hf_mistral_small_3.1_24b_login.sh > logs/hf_dl_mistral_login.nohup.out 2>&1 &
#
set -euo pipefail

REAL_HOME="${HOME}"
SCRATCH_ROOT="${JEVAL_SCRATCH:-/scratch/${USER}}"
WORKDIR="${JEVAL_WORKDIR:-${SCRATCH_ROOT}/Jeval-scratch}"
_JEVAL_VENV="${JEVAL_VENV:-${SCRATCH_ROOT}/venvs/vllm}"

cd "$WORKDIR"
mkdir -p logs

if [ ! -f "$_JEVAL_VENV/bin/activate" ]; then
  echo "ERROR: venv not found at $_JEVAL_VENV" >&2
  exit 1
fi
source "$_JEVAL_VENV/bin/activate"

export XDG_CACHE_HOME="${SCRATCH_ROOT}/.cache"
export HF_HOME="${XDG_CACHE_HOME}/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}"
mkdir -p "$HF_HOME"

unset HF_HUB_OFFLINE
unset HF_DATASETS_OFFLINE

MODEL_ID="${JEVAL_VLLM_MODEL_ID:-mistralai/Mistral-Small-3.1-24B-Instruct-2503}"

echo "=== HF download (login node) ==="
echo "Start: $(date)"
echo "HF_HOME=$HF_HOME"
echo "Model: $MODEL_ID"

if command -v hf >/dev/null 2>&1; then
  hf download "$MODEL_ID"
else
  huggingface-cli download "$MODEL_ID"
fi

echo ""
echo "=== Snapshot dir ==="
ls -la "${HF_HOME}/hub/models--mistralai--Mistral-Small-3.1-24B-Instruct-2503/snapshots" || true

echo ""
echo "=== Prefetch AMA-Bench test split ==="
python3 - <<'PY'
from datasets import load_dataset
load_dataset("AMA-bench/AMA-bench", split="test")
print("AMA-bench test split cached.")
PY

echo "=== Done: $(date) ==="
