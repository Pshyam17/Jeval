#!/bin/bash
# Install torch + vLLM into a venv on scratch/home — NOT on the login node.
#
# Why Slurm (short partition, lots of RAM): `pip install vllm` unpacks huge wheels and
# the login shell gets OOM-killed. GPU compute nodes often have no outbound PyPI, so
# installs belong here; *inference* stays on GPU jobs (`test_episode_gpu.sh`, etc.).
#
# From repo root:
#   sbatch slurm/install_vllm.sh
#
# Override install location:
#   sbatch --export=ALL,JEVAL_VENV=/path/to/venv slurm/install_vllm.sh
#
#SBATCH --job-name=install-vllm
#SBATCH --account=cs6140.202630
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=200G
#SBATCH --time=04:00:00
#SBATCH --output=logs/install_vllm_%j.out
#SBATCH --error=logs/install_vllm_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$HOME/jeval/Jeval-1}"
mkdir -p logs

# Scratch (or a dedicated dir under $HOME) — keeps wheels and venv off tiny login /tmp
ROOT="${SCRATCH:-$HOME/scratch}"
mkdir -p "$ROOT"
export TMPDIR="${ROOT}/pip-tmp-${SLURM_JOB_ID:-local}"
export PIP_CACHE_DIR="${ROOT}/pip-cache"
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR"

VENV="${JEVAL_VENV:-$ROOT/jeval-gpu-venv}"

module purge
module load miniconda3/24.11.1

echo "=== vLLM venv install ==="
echo "node=$SLURMD_NODENAME  start=$(date)"
echo "ROOT=$ROOT"
echo "VENV=$VENV"
echo "TMPDIR=$TMPDIR"

if [ ! -d "$VENV" ]; then
  python3 -m venv "$VENV"
fi
# shellcheck source=/dev/null
source "$VENV/bin/activate"

python3 -m pip install -U pip setuptools wheel
python3 -m pip install "torch==2.4.0" "torchvision==0.19.0" --index-url https://download.pytorch.org/whl/cu121
# Pin vLLM; do NOT use bare `pip install vllm` (pulls latest + giant torch stack).
python3 -m pip install "vllm==0.6.3"

python3 -c "import torch; print('torch', torch.__version__)"
python3 -c "import vllm; print('vllm', getattr(vllm, '__version__', 'ok'))" || true

echo ""
echo "=== DONE $(date) ==="
echo "GPU jobs will auto-activate this venv if it exists at:"
echo "  $VENV"
echo "(see slurm/test_episode_gpu.sh — same default path as ROOT/jeval-gpu-venv)"
