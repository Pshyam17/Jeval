#!/bin/bash
# Run on Explorer LOGIN after rsync'ing laptop-built wheels (torch + vllm wheel-only).
# Downloads the rest of vLLM's dependency tree into the same directory (uses sdists OK).
#
#   export JEVAL_WHEEL_DIR=$HOME/scratch/jeval-pip-wheels   # or your rsync target
#   bash slurm/merge_vllm_wheel_deps_on_login.sh
#
# Uses scratch for TMPDIR to reduce login-node OOM during large metadata work.

set -euo pipefail

ROOT="${SCRATCH:-$HOME/scratch}"
WHEEL_DIR="${JEVAL_WHEEL_DIR:-$ROOT/jeval-pip-wheels}"
TORCH_CUDA_INDEX="${TORCH_CUDA_INDEX:-https://download.pytorch.org/whl/cu121}"
PYPI="https://pypi.org/simple"

export TMPDIR="${ROOT}/pip-tmp-merge-$$"
mkdir -p "$TMPDIR" "$WHEEL_DIR"

module purge 2>/dev/null || true
module load miniconda3/24.11.1

echo "=== merge vLLM deps into $WHEEL_DIR ==="
echo "node=${SLURMD_NODENAME:-login}  start=$(date)"
if [ ! -f "$WHEEL_DIR"/vllm-0.6.3*.whl ]; then
  echo "ERROR: vllm wheel missing in $WHEEL_DIR (rsync laptop wheels first?)" >&2
  exit 1
fi

python3 -m pip download -d "$WHEEL_DIR" "vllm==0.6.3" \
  --find-links "$WHEEL_DIR" \
  --index-url "$TORCH_CUDA_INDEX" \
  --extra-index-url "$PYPI"

echo "=== DONE $(date) ==="
echo "Next: sbatch slurm/install_vllm_offline.sh"
