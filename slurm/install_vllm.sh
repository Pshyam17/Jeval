#!/bin/bash
# Install torch + vLLM into a venv on scratch — same layout many people use, e.g.
#   /scratch/<your_netid>/venvs/vllm   (when $SCRATCH is set to /scratch/<netid>)
# Override path: JEVAL_VENV=/path/to/venv sbatch ...
#
# NOT on the login node (OOM): use this Slurm script or your usual interactive GPU install.
#
# Why Slurm (short partition, lots of RAM): `pip install vllm` unpacks huge wheels and
# the login shell gets OOM-killed.
#
# If short/GPU nodes have NO outbound internet, use the two-step offline path instead:
#   1) Login:  bash slurm/download_vllm_wheels.sh
#   2) Slurm:  sbatch slurm/install_vllm_offline.sh
#
# From repo root — only if THIS NODE can reach pypi.org (many sites: short/GPU have NO outbound PyPI;
# the job will exit immediately with a clear error). On Explorer use install_vllm_offline.sh + wheels from login.
#
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

VENV="${JEVAL_VENV:-$ROOT/venvs/vllm}"

module purge
module load miniconda3/24.11.1

if [ -z "${JEVAL_SKIP_PYPI_CHECK:-}" ]; then
  if ! python3 -c "import urllib.request; urllib.request.urlopen('https://pypi.org/simple/pip/', timeout=15)" 2>/dev/null; then
    echo "ERROR: No HTTPS to pypi.org from $(hostname) — this is normal on Explorer short/GPU (offline compute)." >&2
    echo "Use: bash slurm/download_vllm_wheels.sh on LOGIN, then sbatch slurm/install_vllm_offline.sh" >&2
    echo "Or: curl CUDA wheels on login + sbatch slurm/install_torch_from_local_wheels.sh, then finish vLLM from a full wheel dir." >&2
    echo "Bypass (only if pip works but urllib check fails): JEVAL_SKIP_PYPI_CHECK=1 sbatch ..." >&2
    exit 1
  fi
fi

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
python3 -m pip install --retries 3 --timeout 120 \
  "torch==2.4.0" "torchvision==0.19.0" --index-url https://download.pytorch.org/whl/cu121
# Pin vLLM; do NOT use bare `pip install vllm` (pulls latest + giant torch stack).
python3 -m pip install --retries 3 --timeout 120 "vllm==0.6.3"

python3 -c "import torch; print('torch', torch.__version__)"
python3 -c "import vllm; print('vllm', getattr(vllm, '__version__', 'ok'))" || true

echo ""
echo "=== DONE $(date) ==="
echo "GPU jobs will auto-activate this venv if it exists at:"
echo "  $VENV"
echo "(see slurm/test_episode_gpu.sh — default venv: \$SCRATCH/venvs/vllm)"
