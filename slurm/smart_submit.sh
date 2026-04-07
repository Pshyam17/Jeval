#!/bin/bash
# slurm/smart_submit.sh
# Queries sinfo at submission time, picks the best available GPU tier,
# and injects --partition + --gres into the sbatch call dynamically.
#
# Usage:
#   ./slurm/smart_submit.sh slurm/train_predictor.sh
#   ./slurm/smart_submit.sh slurm/ama_bench_array.sh
#   ./slurm/smart_submit.sh --parsable slurm/train_predictor.sh   (for job chaining)

set -euo pipefail

# Separate any leading sbatch flags (e.g. --parsable, --dependency=...) from the script
SBATCH_PASSTHROUGH=()
TARGET=""
EXTRA=()

for arg in "$@"; do
    if [ -z "$TARGET" ] && [[ "$arg" == -* ]]; then
        SBATCH_PASSTHROUGH+=("$arg")
    elif [ -z "$TARGET" ]; then
        TARGET="$arg"
    else
        EXTRA+=("$arg")
    fi
done

if [ -z "$TARGET" ]; then
    echo "Usage: $0 [sbatch-flags] <script.sh> [extra args...]" >&2
    exit 1
fi

# Tier preference list: "partition|gres|sinfo_grep_pattern|label"
# Checked in order — first tier with idle or mix nodes wins.
TIERS=(
    "gpu|gpu:h200:1|h200|H200"
    "gpu|gpu:a100:1|a100|A100"
    "multigpu|gpu:a100:1|a100|A100-multigpu"
    "gpu-short|gpu:a100:1|a100|A100-short"
    "gpu|gpu:v100:1|v100|V100"
    "gpu|gpu:1|gpu|any-GPU"
)

pick_best() {
    for tier in "${TIERS[@]}"; do
        IFS='|' read -r part gres pattern label <<< "$tier"
        avail=$(sinfo -p "$part" -o "%G %t" --noheader 2>/dev/null \
            | grep -i "$pattern" \
            | grep -cE '\b(idle|mix)\b' || echo 0)
        if [ "${avail:-0}" -gt 0 ]; then
            echo "${part}|${gres}|${label}"
            return 0
        fi
    done
    # Nothing idle — submit anyway to the base gpu partition and wait in queue
    echo "gpu|gpu:1|fallback-any"
}

BEST=$(pick_best)
IFS='|' read -r PART GRES LABEL <<< "$BEST"

echo "[smart_submit] GPU: $LABEL  partition=$PART  gres=$GRES"
echo "[smart_submit] → sbatch ${SBATCH_PASSTHROUGH[*]:-} --partition=$PART --gres=$GRES $TARGET ${EXTRA[*]:-}"

sbatch --account=cs6140.202630 \
       --partition="$PART" \
       --gres="$GRES" \
       "${SBATCH_PASSTHROUGH[@]+"${SBATCH_PASSTHROUGH[@]}"}" \
       "$TARGET" \
       "${EXTRA[@]+"${EXTRA[@]}"}"
