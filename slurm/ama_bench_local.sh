#!/bin/bash
# Option D: AMA-Bench with local Mistral inference on GPU compute nodes
#
# Uses cached Mistral-Small-3.1-24B + vLLM (no internet on compute nodes).
# Prefer:  bash slurm/smart_submit.sh slurm/ama_bench_local.sh
# to target H200→H100→A100→… when idle nodes exist. ~48GB+ VRAM recommended for 24B.
#
#SBATCH --job-name=jeval-ama-local
#SBATCH --account=cs6140.202630
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --array=0-207%20
#SBATCH --output=logs/ama_local_%A_%a.out
#SBATCH --error=logs/ama_local_%A_%a.err

set -euo pipefail

IDX=$SLURM_ARRAY_TASK_ID
OUT="benchmarks/results/ama_bench_episodes/episode_${IDX}.json"

# Skip if already complete
if [ -f "$OUT" ] && python3 -c "
import json, sys
try:
    d = json.load(open('$OUT'))
    sys.exit(0 if 'score' in d else 1)
except:
    sys.exit(1)
" 2>/dev/null; then
    echo "Episode $IDX already complete — skipping"
    exit 0
fi

WORKDIR="${HOME}/jeval/Jeval-1"
cd "$WORKDIR"

module purge
module load miniconda3/24.11.1
module load cuda/12.3 2>/dev/null || true

_JEVAL_VENV_ROOT="${SCRATCH:-$HOME/scratch}"
_JEVAL_VENV="${JEVAL_VENV:-$_JEVAL_VENV_ROOT/venvs/vllm}"
if [ -f "$_JEVAL_VENV/bin/activate" ]; then
  # shellcheck source=/dev/null
  source "$_JEVAL_VENV/bin/activate"
fi

export PYTHONPATH="$WORKDIR:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export JEVAL_INFERENCE_MODE=local
export JEVAL_LOCAL_LLM_BACKEND="${JEVAL_LOCAL_LLM_BACKEND:-vllm}"
export JEVAL_VLLM_MODEL_FORMAT="${JEVAL_VLLM_MODEL_FORMAT:-hf}"

# Model path (cached Mistral)
MODEL_PATH="$HOME/.cache/huggingface/models--mistralai--Mistral-Small-3.1-24B-Instruct-2503/snapshots/68faf511d618ef198fef186659617cfd2eb8e33a"

if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: Model not found at $MODEL_PATH" >&2
    exit 1
fi

echo "=== Episode $IDX  node=$SLURMD_NODENAME  GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)  start=$(date) ==="

python3 benchmarks/run_ama_episode_local.py \
    --episode-idx "$IDX" \
    --dataset AMA-bench/AMA-bench \
    --split test \
    --predictor checkpoints/predictor_v2_best.pt \
    --model-path "$MODEL_PATH" \
    --out "$OUT"

echo "=== Episode $IDX done: $(date) ==="
