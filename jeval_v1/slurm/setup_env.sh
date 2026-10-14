#!/bin/bash
# SLURM script for setting up jeval environment on cluster nodes.
# Also works as a reference for manual setup (see comments below).
#
# Usage:
#   sbatch slurm/setup_env.sh    # Run as SLURM job
#   source slurm/setup_env.sh    # Source for manual setup (skip SBATCH lines)
#
#SBATCH --job-name=jeval-setup
#SBATCH --account=cs6140.202630
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/setup_%j.out
#SBATCH --error=logs/setup_%j.err

set -e
echo "=== jeval env setup ==="
echo "Node: $SLURMD_NODENAME  Start: $(date)"

WORKDIR="${JEVAL_WORKDIR:-$HOME/jeval/Jeval-1}"
mkdir -p "$WORKDIR/logs" "$WORKDIR/checkpoints" \
         "$WORKDIR/benchmarks/results/ama_bench_episodes" \
         "$WORKDIR/train/data"

module purge
module load miniconda3/24.11.1

pip install --quiet --upgrade pip

pip install --quiet \
    torch==2.4.1 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

pip install --quiet \
    sentence-transformers==3.0.1 \
    transformers==4.41.2 \
    datasets==2.20.0 \
    accelerate==0.33.0 \
    numpy==1.26.4 \
    scikit-learn \
    tiktoken \
    openai \
    pytest

python - <<'EOF'
from sentence_transformers import SentenceTransformer
print("Downloading all-mpnet-base-v2 ...")
SentenceTransformer("all-mpnet-base-v2")
print("Done.")
EOF

pip install --quiet -e "$WORKDIR"

# Set fair comparison environment defaults
# These can be overridden in your shell profile or job scripts
cat <<'ENVEOF'

=== Fair Comparison Defaults (add to your shell profile) ===
export JEVAL_ANSWER_MODEL="qwen/qwen3.5-122b-a10b"
export JEVAL_JUDGE_MODEL="qwen/qwen3.5-122b-a10b"
export JEVAL_AMA_DOMAIN="SOFTWARE"
export JEVAL_WORKDIR="$HOME/jeval/Jeval-1"
# Add your NVIDIA API key:
# export NVIDIA_API_KEY="nvapi-..."
ENVEOF

echo ""
echo "=== setup complete: $(date) ==="
