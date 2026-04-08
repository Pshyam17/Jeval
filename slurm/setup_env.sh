#!/bin/bash
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

WORKDIR="$HOME/jeval/Jeval-1"
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

echo "=== setup complete: $(date) ==="
