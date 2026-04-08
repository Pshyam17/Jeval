#!/bin/bash
set -euo pipefail

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
    echo "usage: $0 [sbatch-flags] <script.sh> [extra args...]" >&2
    exit 1
fi

# tiers ordered best→worst: idle beats mix, h200 beats a100 beats v100
# format: partition|gres|sinfo_grep|label
TIERS=(
    "gpu|gpu:h200:1|h200|H200-idle"
    "gpu|gpu:a100:1|a100|A100-idle"
    "multigpu|gpu:a100:1|a100|A100-multigpu-idle"
    "gpu-short|gpu:a100:1|a100|A100-short-idle"
    "gpu|gpu:v100:1|v100|V100-idle"
    "gpu|gpu:h200:1|h200|H200-mix"
    "gpu|gpu:a100:1|a100|A100-mix"
    "gpu|gpu:v100:1|v100|V100-mix"
    "gpu|gpu:1|gpu|any-GPU"
)

pick_best() {
    local best_part="" best_gres="" best_label=""
    local best_score=99

    for tier in "${TIERS[@]}"; do
        IFS='|' read -r part gres pattern label <<< "$tier"

        idle_count=$(sinfo -p "$part" -o "%G %t" --noheader 2>/dev/null \
            | grep -i "$pattern" | grep -c '\bidle\b' || echo 0)
        mix_count=$(sinfo -p "$part" -o "%G %t" --noheader 2>/dev/null \
            | grep -i "$pattern" | grep -c '\bmix\b' || echo 0)

        if [ "${idle_count:-0}" -gt 0 ] && [ "$best_score" -gt 0 ]; then
            best_part="$part"; best_gres="$gres"; best_label="$label"
            best_score=0
        elif [ "${mix_count:-0}" -gt 0 ] && [ "$best_score" -gt 1 ]; then
            best_part="$part"; best_gres="$gres"; best_label="$label (mix)"
            best_score=1
        fi
    done

    if [ -n "$best_part" ]; then
        echo "${best_part}|${best_gres}|${best_label}"
    else
        echo "gpu|gpu:1|fallback-queue"
    fi
}

BEST=$(pick_best)
IFS='|' read -r PART GRES LABEL <<< "$BEST"

echo "[smart_submit] gpu: $LABEL  partition=$PART  gres=$GRES"
echo "[smart_submit] → sbatch ${SBATCH_PASSTHROUGH[*]:-} --partition=$PART --gres=$GRES $TARGET ${EXTRA[*]:-}"

sbatch --account=cs6140.202630 \
       --partition="$PART" \
       --gres="$GRES" \
       "${SBATCH_PASSTHROUGH[@]+"${SBATCH_PASSTHROUGH[@]}"}" \
       "$TARGET" \
       "${EXTRA[@]+"${EXTRA[@]}"}"
