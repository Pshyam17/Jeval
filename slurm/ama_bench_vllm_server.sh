#!/bin/bash
# Full AMA-Bench run using one in-job vLLM server.
# Starts vLLM once, loops episodes START_EP..END_EP, skips completed ones.
#
# Submit (3 batches to fit 8h GPU limit):
#   sbatch --export=ALL,JEVAL_VENV=...,JEVAL_VLLM_MODEL_PATH=...,START_EP=0,END_EP=89   slurm/ama_bench_vllm_server.sh
#   sbatch --export=ALL,JEVAL_VENV=...,JEVAL_VLLM_MODEL_PATH=...,START_EP=90,END_EP=179  slurm/ama_bench_vllm_server.sh
#   sbatch --export=ALL,JEVAL_VENV=...,JEVAL_VLLM_MODEL_PATH=...,START_EP=180,END_EP=207 slurm/ama_bench_vllm_server.sh
#
#SBATCH --job-name=jeval-ama-vllm
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=07:30:00
#SBATCH --output=logs/ama_vllm_%j.out
#SBATCH --error=logs/ama_vllm_%j.err

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
mkdir -p logs benchmarks/results/ama_bench_episodes checkpoints

module purge
module load miniconda3/24.11.1
module load cuda/12.3 2>/dev/null || true

# Activate venv
if [ -n "${JEVAL_VENV:-}" ]; then
  _JEVAL_VENV="$JEVAL_VENV"
elif [ -f "${REAL_HOME}/scratch/jeval-gpu-venv/bin/activate" ]; then
  _JEVAL_VENV="${REAL_HOME}/scratch/jeval-gpu-venv"
elif [ -f "${SCRATCH_ROOT}/jeval-gpu-venv/bin/activate" ]; then
  _JEVAL_VENV="${SCRATCH_ROOT}/jeval-gpu-venv"
else
  echo "ERROR: venv not found. Set JEVAL_VENV." >&2; exit 1
fi
source "$_JEVAL_VENV/bin/activate"

export XDG_CACHE_HOME="${SCRATCH_ROOT}/.cache"
export HF_HOME="${XDG_CACHE_HOME}/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}"
export TRITON_CACHE_DIR="${XDG_CACHE_HOME}/triton"
export TORCH_HOME="${XDG_CACHE_HOME}/torch"
mkdir -p "$XDG_CACHE_HOME" "$HF_HOME" "$TRITON_CACHE_DIR" "$TORCH_HOME"

export NO_PROXY=localhost,127.0.0.1
export no_proxy=localhost,127.0.0.1
export PYTHONPATH="$WORKDIR:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

# Episode range (override via --export)
START_EP="${START_EP:-0}"
END_EP="${END_EP:-207}"

MODEL_ID="${JEVAL_VLLM_MODEL_ID:-mistralai/Mistral-Small-3.1-24B-Instruct-2503}"
MODEL_PATH="${JEVAL_VLLM_MODEL_PATH:-}"

if [ -z "$MODEL_PATH" ] || [ ! -d "$MODEL_PATH" ]; then
  echo "ERROR: MODEL_PATH not found. Set JEVAL_VLLM_MODEL_PATH." >&2; exit 1
fi

PORT="${JEVAL_VLLM_PORT:-$((9000 + SLURM_JOB_ID % 1000))}"
MAX_LEN="${JEVAL_VLLM_MAX_LEN:-8192}"

echo "============================================"
echo "Jeval AMA-Bench Full Run (vLLM server)"
echo "Node:     $SLURMD_NODENAME"
echo "GPU:      $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
echo "Venv:     $_JEVAL_VENV"
echo "Model:    $MODEL_ID"
echo "Episodes: $START_EP to $END_EP"
echo "Port:     $PORT"
echo "Start:    $(date)"
echo "============================================"

# GPU compute capability check
EXTRA_FLAGS=()
CC_MAJOR="$(python3 - <<'PY'
import torch
if torch.cuda.is_available():
    print(torch.cuda.get_device_capability(0)[0])
else:
    print(0)
PY
)"
if [ "${CC_MAJOR:-0}" -lt 8 ]; then
  EXTRA_FLAGS+=(--enforce-eager --dtype float16)
fi

# Start vLLM server
LOG_VLLM="logs/vllm_server_${SLURM_JOB_ID}.log"
setsid vllm serve "$MODEL_PATH" \
  --served-model-name "$MODEL_ID" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --max-model-len "$MAX_LEN" \
  --gpu-memory-utilization 0.90 \
  --limit-mm-per-prompt image=0 \
  "${EXTRA_FLAGS[@]}" >"$LOG_VLLM" 2>&1 &
VLLM_PID=$!

cleanup() {
  set +e
  if kill -0 "$VLLM_PID" 2>/dev/null; then
    kill -TERM "-$VLLM_PID" 2>/dev/null || kill -TERM "$VLLM_PID" 2>/dev/null
    sleep 3
    kill -KILL "-$VLLM_PID" 2>/dev/null || kill -KILL "$VLLM_PID" 2>/dev/null
  fi
}
trap cleanup EXIT

echo "Waiting for vLLM readiness..."
READY=0
for _ in $(seq 1 180); do
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "ERROR: vLLM died during startup" >&2
    tail -60 "$LOG_VLLM" >&2 || true
    exit 1
  fi
  if curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
    READY=1; break
  fi
  sleep 5
done
if [ "$READY" -ne 1 ]; then
  echo "ERROR: vLLM timeout" >&2; tail -60 "$LOG_VLLM" >&2; exit 1
fi
echo "vLLM ready. $(date)"

export JEVAL_NIM_BASE_URL="http://127.0.0.1:${PORT}/v1"
export JEVAL_NIM_MODEL="$MODEL_ID"
export NVIDIA_API_KEY="${NVIDIA_API_KEY:-local-vllm}"

_PRED="${JEVAL_PREDICTOR:-checkpoints/predictor_v2_best.pt}"
_PRED_ARG=()
if [ -f "$_PRED" ]; then
  _PRED_ARG=(--predictor "$_PRED")
fi

COMPLETED=0; SKIPPED=0; FAILED=0

for IDX in $(seq "$START_EP" "$END_EP"); do
  OUT="benchmarks/results/ama_bench_episodes/episode_${IDX}.json"

  # Skip if already has a valid score
  if [ -f "$OUT" ] && python3 -c "
import json,sys
d=json.load(open('$OUT'))
sys.exit(0 if 'score' in d else 1)
" 2>/dev/null; then
    echo "[$(date +%H:%M:%S)] ep $IDX: skip (done)"
    SKIPPED=$((SKIPPED+1))
    continue
  fi

  echo "[$(date +%H:%M:%S)] ep $IDX: starting..."

  python3 benchmarks/run_ama_episode.py \
    --episode-idx "$IDX" \
    --dataset AMA-bench/AMA-bench \
    --split test \
    "${_PRED_ARG[@]}" \
    --out "$OUT" && \
  SCORE=$(python3 -c "import json; print(json.load(open('$OUT')).get('score','?'))" 2>/dev/null || echo "?") && \
  echo "[$(date +%H:%M:%S)] ep $IDX: done score=$SCORE" && \
  COMPLETED=$((COMPLETED+1)) || \
  { echo "[$(date +%H:%M:%S)] ep $IDX: FAILED"; FAILED=$((FAILED+1)); }
done

echo ""
echo "============================================"
echo "Done: completed=$COMPLETED skipped=$SKIPPED failed=$FAILED"
echo "End: $(date)"
echo "============================================"
