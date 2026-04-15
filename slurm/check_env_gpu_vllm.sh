#!/bin/bash
# Verify GPU + vLLM runtime env before scheduling eval.
#
# Usage:
#   bash slurm/smart_submit.sh slurm/check_env_gpu_vllm.sh
#
#SBATCH --job-name=jeval-envcheck
#SBATCH --account=cs6140.202630
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=logs/envcheck_%j.out
#SBATCH --error=logs/envcheck_%j.err

set -euo pipefail

WORKDIR="${HOME}/jeval/Jeval-1"
cd "$WORKDIR"
mkdir -p logs

module purge
module load miniconda3/24.11.1
module load cuda/12.3 2>/dev/null || true

_JEVAL_VENV_ROOT="${SCRATCH:-$HOME/scratch}"
_JEVAL_VENV="${JEVAL_VENV:-$_JEVAL_VENV_ROOT/venvs/vllm}"
if [ -f "$_JEVAL_VENV/bin/activate" ]; then
  source "$_JEVAL_VENV/bin/activate"
else
  echo "ERROR: venv missing at $_JEVAL_VENV"
  exit 1
fi

export JEVAL_INFERENCE_MODE=local
export JEVAL_LOCAL_LLM_BACKEND="${JEVAL_LOCAL_LLM_BACKEND:-vllm}"
export JEVAL_VLLM_MODEL_FORMAT="${JEVAL_VLLM_MODEL_FORMAT:-hf}"

MODEL_PATH="${JEVAL_VLLM_MODEL_PATH:-$HOME/.cache/huggingface/models--mistralai--Mistral-Small-3.1-24B-Instruct-2503/snapshots/68faf511d618ef198fef186659617cfd2eb8e33a}"

echo "=== Jeval env check ==="
echo "node=$SLURMD_NODENAME"
echo "python=$(which python3)"
echo "venv=$_JEVAL_VENV"
echo "JEVAL_INFERENCE_MODE=$JEVAL_INFERENCE_MODE"
echo "JEVAL_LOCAL_LLM_BACKEND=$JEVAL_LOCAL_LLM_BACKEND"
echo "JEVAL_VLLM_MODEL_FORMAT=$JEVAL_VLLM_MODEL_FORMAT"
echo "MODEL_PATH=$MODEL_PATH"
echo "GPU=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"

if [ ! -d "$MODEL_PATH" ]; then
  echo "ERROR: model path missing: $MODEL_PATH"
  exit 1
fi

python3 - <<'PY'
import importlib
mods = ["torch", "vllm", "sentence_transformers", "openai", "datasets"]
for m in mods:
    mod = importlib.import_module(m)
    print(f"{m}: {getattr(mod, '__version__', 'ok')}")
PY

python3 - <<'PY'
import torch
print("cuda_available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA not available")
print("device:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))
PY

echo "ENV_CHECK_OK"
