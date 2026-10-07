#!/bin/bash
# Polls sinfo every POLL_SEC until an A100/H200 node is idle or mixed, then submits the
# LoCoMo compress -> folds -> QA chain pinned to that GPU type and exits.
# Export the JEVAL_* variables first, then:
#   nohup bash jeval_v2/slurm/watch_and_grab_locomo.sh > watcher.log 2>&1 &
set -euo pipefail
: "${JEVAL_REPO_ROOT:?Set JEVAL_REPO_ROOT to the absolute repository root}"
: "${JEVAL_ACCOUNT:?Set JEVAL_ACCOUNT}"
POLL_SEC="${POLL_SEC:-60}"

# partition|sinfo gres pattern|--gres value, in priority order
WATCH_TIERS=(
    "gpu|h200|gpu:h200:1"
    "gpu|a100|gpu:a100:1"
)

log() { echo "[$(date '+%H:%M:%S')] $*"; }

find_idle_tier() {
    for tier in "${WATCH_TIERS[@]}"; do
        IFS='|' read -r part pattern gres <<< "$tier"
        count=$(timeout 5 sinfo -p "$part" -o "%G %t" --noheader 2>/dev/null \
            | grep -i "$pattern" | grep -cE '\b(idle|mix)\b' || true)
        if [ "$count" -gt 0 ]; then
            echo "$part|$gres"
            return 0
        fi
    done
    return 1
}

log "watching for an idle GPU every ${POLL_SEC}s"
while true; do
    if tier=$(find_idle_tier); then
        IFS='|' read -r part gres <<< "$tier"
        log "idle node found: partition=${part} gres=${gres}; submitting"
        export JEVAL_GPU_PARTITION="$part" SBATCH_GRES="$gres"
        bash "${JEVAL_REPO_ROOT}/jeval_v2/slurm/run_locomo.sh"
        log "submitted; watcher done"
        exit 0
    fi
    log "no idle A100/H200; checking again in ${POLL_SEC}s"
    sleep "$POLL_SEC"
done
