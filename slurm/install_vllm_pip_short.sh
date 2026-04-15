#!/bin/bash
# vLLM via pip on a high-RAM short node — ONLY works where compute can reach pypi.org.
#
# On Northeastern Explorer, short nodes typically CANNOT reach PyPI (connection timeouts / endless retries).
# This script exits immediately in that case. Use install_vllm_offline.sh + wheels built on the login node,
# or a pre-built $SCRATCH/venvs/vllm.
#
#   sbatch slurm/install_vllm_pip_short.sh
#
# Override check (rare): JEVAL_SKIP_PYPI_CHECK=1 sbatch ...
#
#SBATCH --job-name=install-vllm-pip
#SBATCH --account=cs6140.202630
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=200G
#SBATCH --time=04:00:00
#SBATCH --output=logs/install_vllm_pip_%j.out
#SBATCH --error=logs/install_vllm_pip_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$HOME/jeval/Jeval-1}"
mkdir -p logs

ROOT="${SCRATCH:-$HOME/scratch}"
V="${JEVAL_VENV:-$ROOT/venvs/vllm}"
export TMPDIR="${ROOT}/pip-tmp-vllm-${SLURM_JOB_ID}"
export PIP_CACHE_DIR="${ROOT}/pip-cache"
mkdir -p "$TMPDIR"

module purge
module load miniconda3/24.11.1
# shellcheck source=/dev/null
source "$V/bin/activate"

echo "node=$SLURMD_NODENAME VENV=$V"

if [ -z "${JEVAL_SKIP_PYPI_CHECK:-}" ]; then
  if ! python3 -c "import urllib.request; urllib.request.urlopen('https://pypi.org/simple/vllm/', timeout=15)" 2>/dev/null; then
    echo "ERROR: No HTTPS to pypi.org from $(hostname). Pip would only retry until the job hits the walltime." >&2
    echo "On Explorer, use slurm/install_vllm_offline.sh after populating ~/scratch/jeval-pip-wheels on LOGIN, or share a working venv." >&2
    exit 1
  fi
fi

python3 -m pip install --retries 2 --timeout 120 "vllm==0.6.3"
python3 -c "import vllm; print('vllm', getattr(vllm,'__version__','ok'))"
echo "=== vLLM OK $(date) ==="
