#!/bin/bash
#SBATCH --job-name=jeval-train
#SBATCH --account=cs6140.202630
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

set -e
echo "=== jeval predictor training ==="
echo "Node: $SLURMD_NODENAME  GPU: $CUDA_VISIBLE_DEVICES  Start: $(date)"

WORKDIR="$HOME/jeval/Jeval-1"
cd "$WORKDIR"

module purge
module load miniconda3/24.11.1

export PYTHONPATH="$WORKDIR:$PYTHONPATH"
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

PAIRS="train/data/pairs_v2.jsonl"
if [ ! -f "$PAIRS" ]; then
    python train/generate_pairs.py \
        --n-faithful      5000 \
        --n-hard-negative 5000 \
        --swebench-pairs  500 \
        --out "$PAIRS"
fi

python train/train_predictor_v2.py \
    --pairs      "$PAIRS" \
    --out-path   checkpoints/predictor_v2_best.pt \
    --lr         3e-4 1e-4 \
    --epochs     100 \
    --batch-size 32

echo "=== training complete: $(date) ==="
echo "Best checkpoint: $(ls -lh checkpoints/predictor_v2_best.pt)"
