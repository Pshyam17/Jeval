#!/bin/bash
# polls sinfo every POLL_SEC until a gpu node goes idle, then submits the
# full train -> ama array -> aggregate pipeline and exits
# run with: nohup bash slurm/watch_and_grab.sh > logs/watcher.log 2>&1 &

set -euo pipefail

WORKDIR="$HOME/jeval/Jeval-1"
POLL_SEC=60
LOG="$WORKDIR/logs/watcher.log"
mkdir -p "$WORKDIR/logs"

# tiers to watch - same priority order as smart_submit, idle only
WATCH_TIERS=(
    "gpu|h200|H200"
    "gpu|a100|A100"
    "gpu-short|a100|A100-short"
    "gpu|v100|V100"
)

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

find_idle_tier() {
    for tier in "${WATCH_TIERS[@]}"; do
        IFS='|' read -r part pattern label <<< "$tier"
        count=$(timeout 5 sinfo -p "$part" -o "%G %t" --noheader 2>/dev/null \
            | grep -i "$pattern" | grep -E '\bidle\b' | wc -l)
        if [ "$count" -gt 0 ]; then
            echo "$part|$pattern|$label"
            return 0
        fi
    done
    return 1
}

log "watcher started — polling every ${POLL_SEC}s for idle gpu node"
log "will submit: train -> ama_bench_array -> aggregate"

cd "$WORKDIR"

while true; do
    if find_idle_tier > /dev/null 2>&1; then
        TIER=$(find_idle_tier)
        IFS='|' read -r PART PATTERN LABEL <<< "$TIER"
        log "idle $LABEL node found on partition=$PART — submitting now"

        TRAIN_JID=$(bash slurm/smart_submit.sh --parsable slurm/train_predictor.sh)
        log "train job submitted: $TRAIN_JID"

        AMA_JID=$(sbatch --parsable \
            --account=cs6140.202630 \
            --dependency=afterok:"$TRAIN_JID" \
            slurm/ama_bench_array.sh)
        log "ama array submitted: $AMA_JID (depends on $TRAIN_JID)"

        AGG_JID=$(sbatch --parsable \
            --account=cs6140.202630 \
            --dependency=afterok:"$AMA_JID" \
            slurm/aggregate_ama_results.sh)
        log "aggregate submitted: $AGG_JID (depends on $AMA_JID)"

        log "pipeline queued: train=$TRAIN_JID  array=$AMA_JID  agg=$AGG_JID"
        log "watcher done — exiting"
        exit 0
    fi

    log "no idle gpu found — checking again in ${POLL_SEC}s"
    sleep "$POLL_SEC"
done
