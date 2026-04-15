#!/bin/bash
# Finish torch+torchvision install from wheels on disk (no PyPI). Login nodes often OOM
# unpacking CUDA stacks; short nodes have RAM but usually no internet.
#
# Prereq: curl wheels on login into $SCRATCH/manual-wheels/ (see install_vllm.sh header).
#
#   sbatch slurm/install_torch_from_local_wheels.sh
#
#SBATCH --job-name=install-torch-local
#SBATCH --account=cs6140.202630
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --output=logs/install_torch_local_%j.out
#SBATCH --error=logs/install_torch_local_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$HOME/jeval/Jeval-1}"
mkdir -p logs

ROOT="${SCRATCH:-$HOME/scratch}"
W="$ROOT/manual-wheels"
V="${JEVAL_VENV:-$ROOT/venvs/vllm}"

module purge
module load miniconda3/24.11.1

# shellcheck source=/dev/null
source "$V/bin/activate"

echo "node=$SLURMD_NODENAME VENV=$V W=$W"
ls -la "$W"/*.whl

# Deps (nvidia-*, triton, …) should already be in the venv from the partial login install;
# avoid --no-index so pip can fill any missing small wheels if the node has PyPI (optional).
python3 -m pip install --no-deps \
  "$W"/torch-2.4.0+cu121-cp310-cp310-linux_x86_64.whl \
  "$W"/torchvision-0.19.0+cu121-cp310-cp310-linux_x86_64.whl

python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
echo "=== torch OK $(date) ==="
