#!/bin/bash
#SBATCH --job-name=jeval-aggregate
#SBATCH --account=cs6140.202630
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=logs/aggregate_%j.out
#SBATCH --error=logs/aggregate_%j.err

set -e
echo "=== AMA-Bench aggregate ==="
echo "Node: $SLURMD_NODENAME  Start: $(date)"

WORKDIR="$HOME/jeval/Jeval-1"
cd "$WORKDIR"

module purge
module load miniconda3/24.11.1

export PYTHONPATH="$WORKDIR:$PYTHONPATH"

# Optional: specify domain for stratified results (default: SOFTWARE)
DOMAIN="${JEVAL_AMA_DOMAIN:-SOFTWARE}"
OUT_BASE="benchmarks/results/ama_bench_${DOMAIN,,}_final.json"

python benchmarks/aggregate_ama_results.py \
    --results-dir benchmarks/results/ama_bench_episodes \
    --out         "$OUT_BASE" \
    --domain      "$DOMAIN"

echo "=== aggregate complete: $(date) ==="
