#!/bin/bash
#SBATCH --job-name=jeval-ama
#SBATCH --account=cs6140.202630
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --array=0-207%50
#SBATCH --output=logs/ama_%A_%a.out
#SBATCH --error=logs/ama_%A_%a.err

set -e
IDX=$SLURM_ARRAY_TASK_ID
OUT="benchmarks/results/ama_bench_episodes/episode_${IDX}.json"

if [ -f "$OUT" ] && python -c "
import json, sys
d = json.load(open('$OUT'))
sys.exit(0 if 'score' in d else 1)
" 2>/dev/null; then
    echo "episode $IDX already complete — skipping"
    exit 0
fi

WORKDIR="$HOME/jeval/Jeval-1"
cd "$WORKDIR"

module purge
module load miniconda3/24.11.1

export PYTHONPATH="$WORKDIR:$PYTHONPATH"
export TOKENIZERS_PARALLELISM=false

echo "=== episode $IDX  node=$SLURMD_NODENAME  start=$(date) ==="

python benchmarks/run_ama_episode.py \
    --episode-idx "$IDX" \
    --dataset     AMA-bench/AMA-bench \
    --split       test \
    --predictor   checkpoints/predictor_v2_best.pt \
    --out         "$OUT"

echo "=== episode $IDX done: $(date) ==="
