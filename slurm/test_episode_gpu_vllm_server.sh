#!/bin/bash
# Episode-0 GPU test using one in-job vLLM server (OpenAI-compatible API).
# Mirrors the pattern documented in zzz/docs/vLLM.md:
# start vLLM once -> readiness check -> run benchmark -> clean shutdown.
#
# Submit:
#   bash slurm/smart_submit.sh slurm/test_episode_gpu_vllm_server.sh
#
# Optional overrides:
#   sbatch --export=ALL,JEVAL_VENV=/scratch/.../venvs/vllm,JEVAL_VLLM_MODEL_ID=mistralai/Mistral-Small-3.1-24B-Instruct-2503 slurm/test_episode_gpu_vllm_server.sh
#
#SBATCH --job-name=jeval-ep0-vllm
#SBATCH --account=cs6140.202630
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=01:30:00
#SBATCH --output=logs/test_ep0_vllm_%j.out
#SBATCH --error=logs/test_ep0_vllm_%j.err

set -euo pipefail

WORKDIR="${HOME}/jeval/Jeval-1"
REAL_HOME="$HOME"
cd "$WORKDIR"
mkdir -p logs benchmarks/results/ama_bench_episodes

module purge
module load miniconda3/24.11.1
module load cuda/12.3 2>/dev/null || true

# Activate venv (prefer explicit JEVAL_VENV; fallback to scratch default)
_JEVAL_VENV_ROOT="${SCRATCH:-$HOME/scratch}"
_JEVAL_VENV="${JEVAL_VENV:-$_JEVAL_VENV_ROOT/jeval-gpu-venv}"
if [ ! -f "$_JEVAL_VENV/bin/activate" ]; then
  echo "ERROR: venv not found at $_JEVAL_VENV" >&2
  exit 1
fi
source "$_JEVAL_VENV/bin/activate"

# Keep caches on scratch to avoid home quota and speed up repeated runs.
export HOME="${SCRATCH:-$REAL_HOME/scratch}"
export XDG_CACHE_HOME="${HOME}/.cache"
export HF_HOME="${HOME}/.cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}"
export TRITON_CACHE_DIR="${HOME}/.cache/triton"
export TORCH_HOME="${HOME}/.cache/torch"
mkdir -p "$XDG_CACHE_HOME" "$HF_HOME" "$TRITON_CACHE_DIR" "$TORCH_HOME"

export NO_PROXY=localhost,127.0.0.1
export no_proxy=localhost,127.0.0.1
export PYTHONPATH="$WORKDIR:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

MODEL_PATH="${JEVAL_VLLM_MODEL_PATH:-$REAL_HOME/.cache/huggingface/models--mistralai--Mistral-Small-3.1-24B-Instruct-2503/snapshots/68faf511d618ef198fef186659617cfd2eb8e33a}"
MODEL_ID="${JEVAL_VLLM_MODEL_ID:-mistralai/Mistral-Small-3.1-24B-Instruct-2503}"
PORT="${JEVAL_VLLM_PORT:-$((9000 + SLURM_JOB_ID % 1000))}"
MAX_LEN="${JEVAL_VLLM_MAX_LEN:-8192}"
OUT="benchmarks/results/ama_bench_episodes/episode_0_gpu_test.json"

if [ ! -d "$MODEL_PATH" ]; then
  echo "ERROR: model snapshot not found: $MODEL_PATH" >&2
  exit 1
fi

echo "=========================================="
echo "Jeval AMA-Bench Test (Episode 0, vLLM server)"
echo "Node:  $SLURMD_NODENAME"
echo "GPU:   $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
echo "Venv:  $_JEVAL_VENV"
echo "Model: $MODEL_ID (path: $MODEL_PATH)"
echo "Port:  $PORT"
echo "Start: $(date)"
echo "=========================================="

# V100/T4-style compatibility flags if needed (< sm_80)
EXTRA_FLAGS=()
CC_MAJOR="$(python3 - <<'PY'
import torch
if torch.cuda.is_available():
    cc = torch.cuda.get_device_capability(0)
    print(cc[0])
else:
    print(0)
PY
)"
if [ "${CC_MAJOR:-0}" -lt 8 ]; then
  EXTRA_FLAGS+=(--enforce-eager --dtype float16)
fi

LOG_VLLM="logs/vllm_server_${SLURM_JOB_ID}.log"
setsid vllm serve "$MODEL_PATH" \
  --served-model-name "$MODEL_ID" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --max-model-len "$MAX_LEN" \
  --gpu-memory-utilization 0.90 \
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

echo "Waiting for vLLM readiness on /v1/models ..."
READY=0
for _ in $(seq 1 180); do
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "ERROR: vLLM process died during startup" >&2
    tail -120 "$LOG_VLLM" >&2 || true
    exit 1
  fi
  if curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
    READY=1
    break
  fi
  sleep 5
done
if [ "$READY" -ne 1 ]; then
  echo "ERROR: vLLM did not become ready within timeout" >&2
  tail -120 "$LOG_VLLM" >&2 || true
  exit 1
fi
echo "vLLM ready."

# Point existing NIM/OpenAI client flow to local vLLM API.
export JEVAL_NIM_BASE_URL="http://127.0.0.1:${PORT}/v1"
export JEVAL_NIM_MODEL="$MODEL_ID"
export NVIDIA_API_KEY="${NVIDIA_API_KEY:-local-vllm}"

python3 benchmarks/run_ama_episode.py \
  --episode-idx 0 \
  --dataset AMA-bench/AMA-bench \
  --split test \
  --predictor checkpoints/predictor_v2_best.pt \
  --out "$OUT"

echo "=== Test Complete: $(date) ==="
python3 - <<'PY'
import json
from pathlib import Path
p = Path("benchmarks/results/ama_bench_episodes/episode_0_gpu_test.json")
if not p.exists():
    raise SystemExit("ERROR: result JSON not found")
d = json.loads(p.read_text())
print("Episode:", d.get("episode_id", "N/A"))
print("Score:", d.get("score", 0))
print("Compression:", d.get("original_tokens", 0), "->", d.get("compressed_tokens", 0))
if "error" in d:
    print("ERROR:", d["error"])
PY
