#!/bin/bash
#SBATCH --job-name=jeval-droid
#SBATCH --account=cs6140.202630
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=logs/droid_%j.out
#SBATCH --error=logs/droid_%j.err

set -e
echo "=== DroidBench ==="
echo "Node: $SLURMD_NODENAME  Start: $(date)"

WORKDIR="$HOME/jeval/Jeval-1"
cd "$WORKDIR"

module purge
module load miniconda3/24.11.1

export PYTHONPATH="$WORKDIR:$PYTHONPATH"
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# NVIDIA_API_KEY must be set in the environment or sourced from ~/.jeval_secrets
if [ -z "${NVIDIA_API_KEY:-}" ]; then
    if [ -f "$HOME/.jeval_secrets" ]; then
        # shellcheck source=/dev/null
        source "$HOME/.jeval_secrets"
    fi
fi

if [ -z "${NVIDIA_API_KEY:-}" ]; then
    echo "ERROR: NVIDIA_API_KEY not set. Export it or add it to ~/.jeval_secrets" >&2
    exit 1
fi

mkdir -p benchmarks/results

python jeval/benchmarks/droid_bench.py \
    2>&1 | tee benchmarks/results/droid_bench_raw.txt

# droid_bench.py writes benchmark_results.json at repo root — move it into results/
if [ -f "benchmark_results.json" ]; then
    mv benchmark_results.json benchmarks/results/droid_bench_final.json
    echo "Results saved to benchmarks/results/droid_bench_final.json"
fi

echo "=== DroidBench complete: $(date) ==="
