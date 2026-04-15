#!/bin/bash
# outlines (+ heavy deps like pandas/numba) often OOMs on the login node; try on short with RAM.
# On Explorer this only works if the node can reach PyPI (often it cannot — then use a wheel dir +
# pip install --no-index --find-links, built on login via pip download).
#
#   sbatch slurm/install_outlines_on_short.sh
#
#SBATCH --job-name=install-outlines
#SBATCH --account=cs6140.202630
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --output=logs/install_outlines_%j.out
#SBATCH --error=logs/install_outlines_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$HOME/jeval/Jeval-1}"
mkdir -p logs

ROOT="${SCRATCH:-$HOME/scratch}"
V="${JEVAL_VENV:-$ROOT/venvs/vllm}"
export TMPDIR="${ROOT}/pip-tmp-outl-${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"

module purge
module load miniconda3/24.11.1
# shellcheck source=/dev/null
source "$V/bin/activate"

echo "node=$SLURMD_NODENAME"

if [ -z "${JEVAL_SKIP_PYPI_CHECK:-}" ]; then
  if ! python3 -c "import urllib.request; urllib.request.urlopen('https://pypi.org/simple/outlines/', timeout=15)" 2>/dev/null; then
    echo "ERROR: no PyPI from $(hostname). Stage wheels on LOGIN, then:" >&2
    echo "  pip install --no-index --find-links \$WHEEL_DIR 'outlines>=0.0.43,<0.1'" >&2
    exit 1
  fi
fi

python3 -m pip install --retries 2 --timeout 120 "outlines>=0.0.43,<0.1"
python3 -c "import outlines; print('outlines ok')"
echo "=== DONE $(date) ==="
