#!/bin/bash
# keeps retrying srun --immediate until it grabs a gpu node
# runs training directly (no sbatch queue), then chains ama + aggregate via sbatch
# usage: nohup bash slurm/srun_loop.sh > logs/srun_loop.log 2>&1 &

set -euo pipefail

WORKDIR="$HOME/jeval/Jeval-1"
ACCOUNT="cs6140.202630"
LOG="$WORKDIR/logs/srun_loop.log"
ATTEMPT=0

# gpu tiers to try in order (best → fallback)
TIERS=(
    "gpu|gpu:h200:1"
    "gpu|gpu:h100:1"
    "gpu|gpu:a100:1"
    "gpu-short|gpu:a100:1"
    "gpu|gpu:l40s:1"
    "gpu|gpu:l40:1"
    "gpu|gpu:v100:1"
)

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

cd "$WORKDIR"
mkdir -p logs
log "srun loop started — will retry every 60s until a gpu slot opens"

while true; do
    ATTEMPT=$((ATTEMPT + 1))
    log "attempt $ATTEMPT — trying each gpu tier"

    for tier in "${TIERS[@]}"; do
        IFS='|' read -r part gres <<< "$tier"
        log "  trying partition=$part gres=$gres (immediate=60)"

        if srun --account="$ACCOUNT" \
                --partition="$part" \
                --gres="$gres" \
                --cpus-per-task=8 \
                --mem=64G \
                --time=2:00:00 \
                --immediate=60 \
                bash "$WORKDIR/slurm/train_predictor.sh"; then

            log "training complete — submitting ama + aggregate"

            # find the checkpoint
            CKPT="$WORKDIR/checkpoints/predictor_v2_best.pt"
            if [ ! -f "$CKPT" ]; then
                log "WARNING: checkpoint not found at $CKPT"
            fi

            AMA_JID=$(sbatch --parsable --account="$ACCOUNT" \
                slurm/ama_bench_array.sh)
            AGG_JID=$(sbatch --parsable --account="$ACCOUNT" \
                --dependency=afterok:"$AMA_JID" \
                slurm/aggregate_ama_results.sh)

            log "array=$AMA_JID  agg=$AGG_JID — done"
            exit 0
        else
            log "  $part/$gres busy — trying next tier"
        fi
    done

    log "all tiers busy — waiting 60s before next attempt"
    sleep 60
done
