#!/bin/bash
# Install torch + vLLM from a local wheel directory (NO INTERNET on compute).
# Run on Slurm with high RAM — same reason as install_vllm.sh (large unpack).
#
# Prerequisite: wheels on $SCRATCH from login node:
#   bash slurm/download_vllm_wheels.sh
#
# Usage:
#   sbatch slurm/install_vllm_offline.sh
#
# Env:
#   JEVAL_WHEEL_DIR — default: ${SCRATCH:-$HOME/scratch}/jeval-pip-wheels
#   JEVAL_VENV      — default: ${SCRATCH:-$HOME/scratch}/venvs/vllm
#   JEVAL_WHEEL_EXTRA_PKGS — optional space-separated list installed after core trio (must be present in WHEEL_DIR)
#
#SBATCH --job-name=install-vllm-offline
#SBATCH --account=cs6140.202630
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=200G
#SBATCH --time=04:00:00
#SBATCH --output=logs/install_vllm_offline_%j.out
#SBATCH --error=logs/install_vllm_offline_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$HOME/jeval/Jeval-1}"
mkdir -p logs

ROOT="${SCRATCH:-$HOME/scratch}"
mkdir -p "$ROOT"
export TMPDIR="${ROOT}/pip-tmp-offline-${SLURM_JOB_ID:-local}"
mkdir -p "$TMPDIR"

WHEEL_DIR="${JEVAL_WHEEL_DIR:-$ROOT/jeval-pip-wheels}"
VENV="${JEVAL_VENV:-$ROOT/venvs/vllm}"

module purge
module load miniconda3/24.11.1

echo "=== vLLM OFFLINE venv install ==="
echo "node=$SLURMD_NODENAME  start=$(date)"
echo "ROOT=$ROOT"
echo "WHEEL_DIR=$WHEEL_DIR"
echo "VENV=$VENV"

if [ ! -d "$WHEEL_DIR" ] || [ -z "$(find "$WHEEL_DIR" -maxdepth 1 -name '*.whl' -print -quit)" ]; then
  echo "ERROR: No wheels in $WHEEL_DIR — run on LOGIN: bash slurm/download_vllm_wheels.sh" >&2
  exit 1
fi

if [ ! -d "$VENV" ]; then
  python3 -m venv "$VENV"
fi
# shellcheck source=/dev/null
source "$VENV/bin/activate"

python3 -m pip install --no-index --find-links "$WHEEL_DIR" -U pip setuptools wheel
python3 -m pip install --no-index --find-links "$WHEEL_DIR" "torch==2.4.0" "torchvision==0.19.0"
python3 -m pip install --no-index --find-links "$WHEEL_DIR" "vllm==0.6.3"

if [ -n "${JEVAL_WHEEL_EXTRA_PKGS:-}" ]; then
  echo "--- offline extras: $JEVAL_WHEEL_EXTRA_PKGS ---"
  python3 -m pip install --no-index --find-links "$WHEEL_DIR" $JEVAL_WHEEL_EXTRA_PKGS
fi

python3 -c "import torch; print('torch', torch.__version__)"
python3 -c "import vllm; print('vllm', getattr(vllm, '__version__', 'ok'))" || true

echo ""
echo "=== DONE $(date) ==="
echo "GPU jobs: activate $VENV (see slurm/test_episode_gpu.sh)"
