#!/bin/bash
#SBATCH --job-name=jeval-train
#SBATCH --account=cs6140.202630
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
# --partition and --gres injected by smart_submit.sh at submission time
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

set -e
echo "=== jeval predictor training ==="
echo "Node: $SLURMD_NODENAME  GPU: $CUDA_VISIBLE_DEVICES  Start: $(date)"

WORKDIR="$HOME/jeval/Jeval-1"
cd "$WORKDIR"

module purge
module load miniconda3/24.11.1
source activate jeval

export PYTHONPATH="$WORKDIR:$PYTHONPATH"
export TOKENIZERS_PARALLELISM=false

# Step 1 — generate training pairs (skips if output already exists)
PAIRS="train/data/pairs_v2.jsonl"
if [ ! -f "$PAIRS" ]; then
    echo "--- Generating pairs ---"
    python3.12 train/generate_pairs.py \
        --n-faithful      5000 \
        --n-hard-negative 5000 \
        --swebench-pairs  500 \
        --out "$PAIRS"
else
    echo "Pairs file exists ($PAIRS) — skipping generation"
fi

# Step 2 — train with lr sweep; prints separation ratio after each run
echo "--- Training predictor v2 ---"
python3.12 train/train_predictor_v2.py \
    --pairs      "$PAIRS" \
    --out-path   checkpoints/predictor_v2_best.pt \
    --lr         3e-4 1e-4 \
    --epochs     100 \
    --batch-size 32

echo "=== training complete: $(date) ==="
echo "Best checkpoint: $(ls -lh checkpoints/predictor_v2_best.pt)"
