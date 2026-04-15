#!/bin/bash
# Download torch + vLLM wheels on the LOGIN NODE (needs internet), into $SCRATCH.
# Wheels are Linux x86_64 matching Explorer — safe to install offline on GPU/short nodes.
#
# Usage (SSH as shyamsundar.p, from repo root):
#   module load miniconda3/24.11.1   # if not already loaded
#   cd ~/jeval/Jeval-1 && bash slurm/download_vllm_wheels.sh
#
# Optional — also pull packages for check_env / AMA eval (larger download):
#   JEVAL_WHEEL_EXTRA_PKGS="sentence-transformers openai datasets" bash slurm/download_vllm_wheels.sh
#
# Env:
#   JEVAL_WHEEL_DIR   — default: ${SCRATCH:-$HOME/scratch}/jeval-pip-wheels
#   TORCH_CUDA_INDEX  — default: https://download.pytorch.org/whl/cu121 (must match install_vllm*.sh)

set -euo pipefail

ROOT="${SCRATCH:-$HOME/scratch}"
mkdir -p "$ROOT"
WHEEL_DIR="${JEVAL_WHEEL_DIR:-$ROOT/jeval-pip-wheels}"
TORCH_CUDA_INDEX="${TORCH_CUDA_INDEX:-https://download.pytorch.org/whl/cu121}"
PYPI="https://pypi.org/simple"

mkdir -p "$WHEEL_DIR"
# Keep pip temp/cache on scratch — login /tmp is small; default pip cache can OOM.
export TMPDIR="${JEVAL_PIP_TMPDIR:-$ROOT/pip-tmp-download}"
export PIP_CACHE_DIR="${JEVAL_PIP_CACHE_DIR:-$ROOT/pip-cache-download}"
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR"

if [ -n "${SLURM_JOB_ID:-}" ]; then
  echo "NOTE: Running under Slurm (OK for download_vllm_wheels_job.sh)."
fi

echo "=== download_vllm_wheels ==="
echo "WHEEL_DIR=$WHEEL_DIR"
echo "TORCH_CUDA_INDEX=$TORCH_CUDA_INDEX"
echo "TMPDIR=$TMPDIR"
echo "start=$(date)"
echo ""

python3 -m pip install -q -U pip setuptools wheel
echo "--- pip / setuptools / wheel (for offline venv bootstrap) ---"
python3 -m pip download -d "$WHEEL_DIR" "pip>=24.0" "setuptools>=69.0" "wheel>=0.43.0"

# Split: resolving torch+torchvision+vllm in one shot spikes RAM on login nodes (process killed).
echo "--- torch + torchvision ---"
python3 -m pip download -d "$WHEEL_DIR" \
  "torch==2.4.0" "torchvision==0.19.0" \
  --index-url "$TORCH_CUDA_INDEX" \
  --extra-index-url "$PYPI"

echo "--- vllm (reuse wheels in $WHEEL_DIR) ---"
python3 -m pip download -d "$WHEEL_DIR" \
  "vllm==0.6.3" \
  --index-url "$TORCH_CUDA_INDEX" \
  --extra-index-url "$PYPI" \
  --find-links "$WHEEL_DIR"

if [ -n "${JEVAL_WHEEL_EXTRA_PKGS:-}" ]; then
  echo "--- extras: $JEVAL_WHEEL_EXTRA_PKGS ---"
  # Reuse torch index so any CUDA-aware deps resolve like the online install path.
  python3 -m pip download -d "$WHEEL_DIR" $JEVAL_WHEEL_EXTRA_PKGS \
    --index-url "$TORCH_CUDA_INDEX" \
    --extra-index-url "$PYPI" \
    --find-links "$WHEEL_DIR"
fi

echo ""
echo "=== DONE $(date) ==="
echo "Wheel count: $(find "$WHEEL_DIR" -maxdepth 1 -name '*.whl' | wc -l | tr -d ' ')"
echo "Next (no internet on compute):  sbatch slurm/install_vllm_offline.sh"
echo "Override wheel dir:  sbatch --export=ALL,JEVAL_WHEEL_DIR=$WHEEL_DIR slurm/install_vllm_offline.sh"
