#!/bin/bash
# Test single episode (episode 0) on GPU with local inference (vLLM by default).
#
# Usage (preferred — picks best idle GPU tier H200→H100→A100→…):
#   bash slurm/smart_submit.sh slurm/test_episode_gpu.sh
#
# Direct submit (any single GPU the scheduler assigns):
#   sbatch slurm/test_episode_gpu.sh
#
#SBATCH --job-name=jeval-test-ep0
#SBATCH --account=cs6140.202630
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=00:30:00
#SBATCH --output=logs/test_ep0_%j.out
#SBATCH --error=logs/test_ep0_%j.err

set -euo pipefail

WORKDIR="${HOME}/jeval/Jeval-1"
cd "$WORKDIR"
mkdir -p logs

module purge
module load miniconda3/24.11.1
module load cuda/12.3 2>/dev/null || true

# venv from slurm/install_vllm.sh (scratch or $HOME/scratch) — never rely on login pip
_JEVAL_VENV_ROOT="${SCRATCH:-$HOME/scratch}"
_JEVAL_VENV="${JEVAL_VENV:-$_JEVAL_VENV_ROOT/venvs/vllm}"
if [ -f "$_JEVAL_VENV/bin/activate" ]; then
  # shellcheck source=/dev/null
  source "$_JEVAL_VENV/bin/activate"
  echo "Activated Jeval GPU venv: $_JEVAL_VENV"
else
  echo "WARNING: venv not found at $_JEVAL_VENV — run: sbatch slurm/install_vllm.sh" >&2
fi

export PYTHONPATH="$WORKDIR:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export JEVAL_INFERENCE_MODE=local
# Local completions: vLLM (recommended for Mistral-Small-3.1). Override with --backend hf if needed.
export JEVAL_LOCAL_LLM_BACKEND="${JEVAL_LOCAL_LLM_BACKEND:-vllm}"
# HF snapshot layout on disk; use mistral if your weights are in Mistral's native packaging.
export JEVAL_VLLM_MODEL_FORMAT="${JEVAL_VLLM_MODEL_FORMAT:-hf}"

# Model path
MODEL_PATH="$HOME/.cache/huggingface/models--mistralai--Mistral-Small-3.1-24B-Instruct-2503/snapshots/68faf511d618ef198fef186659617cfd2eb8e33a"

if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: Model not found at $MODEL_PATH" >&2
    exit 1
fi

echo "=========================================="
echo "Jeval AMA-Bench Test (Episode 0)"
echo "=========================================="
echo "Node:     $SLURMD_NODENAME"
echo "GPU:      $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'no GPU detected')"
echo "Model:    Mistral-Small-3.1-24B (cached)"
echo "Start:    $(date)"
echo ""

python3 benchmarks/run_ama_episode_local.py \
    --episode-idx 0 \
    --dataset AMA-bench/AMA-bench \
    --split test \
    --predictor checkpoints/predictor_v2_best.pt \
    --model-path "$MODEL_PATH" \
    --out benchmarks/results/ama_bench_episodes/episode_0_gpu_test.json

echo ""
echo "=========================================="
echo "Test Complete"
echo "=========================================="
echo "End: $(date)"

if [ -f benchmarks/results/ama_bench_episodes/episode_0_gpu_test.json ]; then
    echo ""
    echo "Result:"
    python3 -c "
import json
d = json.load(open('benchmarks/results/ama_bench_episodes/episode_0_gpu_test.json'))
print(f\"  Episode: {d.get('episode_id', 'N/A')}\")
print(f\"  Score: {d.get('score', 0):.2f}\")
print(f\"  Correct: {d.get('correct', 0)}/{d.get('total', 0)}\")
print(f\"  Compression: {d.get('original_tokens', 0)} → {d.get('compressed_tokens', 0)} tokens ({d.get('compression_ratio', 0):.2f})\")
if 'error' in d:
    print(f\"  ERROR: {d['error']}\")
"
else
    echo "ERROR: No output file generated"
    exit 1
fi
