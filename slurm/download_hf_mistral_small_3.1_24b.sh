#!/bin/bash
# One-time (or refresh) download of Mistral Small 3.1 24B into scratch HF cache.
#
# WARNING (Explorer): compute nodes usually have NO outbound internet to huggingface.co.
# If this job fails with "Network is unreachable", use the login-node script instead:
#   bash slurm/download_hf_mistral_small_3.1_24b_login.sh
#
# Run from repo root on the cluster:
#   cd /scratch/$USER/Jeval-scratch && sbatch slurm/download_hf_mistral_small_3.1_24b.sh
#
# Requires Hugging Face access to the model (accept license / HF_TOKEN if gated).
#
#SBATCH --job-name=hf-dl-mistral-small31
# Omit --account to use your Slurm default (override: sbatch --account=YOUR_ACCT ...).
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/hf_dl_mistral_small31_%j.out
#SBATCH --error=logs/hf_dl_mistral_small31_%j.err

set -euo pipefail

REAL_HOME="${HOME}"
SCRATCH_ROOT="${JEVAL_SCRATCH:-${SCRATCH:-/scratch/${USER}}}"

if [ -n "${JEVAL_WORKDIR:-}" ]; then
  WORKDIR="$JEVAL_WORKDIR"
elif [ -d "${REAL_HOME}/jeval/Jeval-1" ]; then
  WORKDIR="${REAL_HOME}/jeval/Jeval-1"
else
  WORKDIR="${SCRATCH_ROOT}/Jeval-scratch"
fi

cd "$WORKDIR"
mkdir -p logs

module purge
module load miniconda3/24.11.1

_JEVAL_VENV="${JEVAL_VENV:-${SCRATCH_ROOT}/venvs/vllm}"
if [ ! -f "$_JEVAL_VENV/bin/activate" ]; then
  echo "ERROR: venv not found at $_JEVAL_VENV" >&2
  exit 1
fi
source "$_JEVAL_VENV/bin/activate"

export XDG_CACHE_HOME="${SCRATCH_ROOT}/.cache"
export HF_HOME="${XDG_CACHE_HOME}/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}"
mkdir -p "$HF_HOME"

# Online download + optional dataset cache for offline eval jobs
unset HF_HUB_OFFLINE
unset HF_DATASETS_OFFLINE
export HF_HUB_ENABLE_HF_TRANSFER=1

MODEL_ID="${JEVAL_VLLM_MODEL_ID:-mistralai/Mistral-Small-3.1-24B-Instruct-2503}"

echo "=== HF download ==="
echo "Node:  $SLURMD_NODENAME"
echo "Start: $(date)"
echo "HF_HOME=$HF_HOME"
echo "Model: $MODEL_ID"
echo "huggingface-cli: $(command -v huggingface-cli)"

huggingface-cli download "$MODEL_ID"

echo ""
echo "=== Snapshot path (for vLLM) ==="
ls -la "${HF_HOME}/hub/models--mistralai--Mistral-Small-3.1-24B-Instruct-2503/snapshots" || true

echo ""
echo "=== Prefetch AMA-Bench test split (for offline episode runs) ==="
python3 - <<'PY'
from datasets import load_dataset
load_dataset("AMA-bench/AMA-bench", split="test")
print("AMA-bench test split cached.")
PY

echo "=== Done: $(date) ==="
