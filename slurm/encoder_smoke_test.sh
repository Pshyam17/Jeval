#!/bin/bash
# Smoke test: load all-mpnet-base-v2 on CPU and run one encode.
# Usage (from repo root, after logs/ exists):
#   mkdir -p logs
#   sbatch slurm/encoder_smoke_test.sh
#
#SBATCH --job-name=jeval-enc-smoke
#SBATCH --account=cs6140.202630
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=logs/encoder_smoke_%j.out
#SBATCH --error=logs/encoder_smoke_%j.err

set -euo pipefail

WORKDIR="${SLURM_SUBMIT_DIR:-$HOME/jeval/Jeval-1}"
cd "$WORKDIR"
mkdir -p logs

module purge
module load miniconda3/24.11.1

export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"

echo "=== encoder smoke  node=$SLURMD_NODENAME  start=$(date) ==="
python -u -c "
from sentence_transformers import SentenceTransformer
print('loading...', flush=True)
m = SentenceTransformer('all-mpnet-base-v2', device='cpu')
print('loaded', flush=True)
out = m.encode(['test sentence'], batch_size=8)
print('encoded:', out.shape, flush=True)
"
echo "=== done $(date) ==="
