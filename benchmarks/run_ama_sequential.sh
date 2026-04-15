#!/bin/bash
# Sequential AMA-Bench runner for login node (Option A: disciplined NIM usage)
#
# Runs episodes one-at-a-time to avoid login node watchdog kills.
# Progress is logged; can resume from last completed episode.
# Run in nohup: nohup bash benchmarks/run_ama_sequential.sh &> logs/ama_sequential.log &
#
# Usage:
#   cd ~/jeval/Jeval-1
#   mkdir -p logs benchmarks/results/ama_bench_episodes
#   nohup bash benchmarks/run_ama_sequential.sh &> logs/ama_sequential.log &
#   tail -f logs/ama_sequential.log

set -euo pipefail

WORKDIR="${WORKDIR:-$HOME/jeval/Jeval-1}"
cd "$WORKDIR"

TOTAL_EPISODES=208
START_IDX="${START_IDX:-0}"
END_IDX="${END_IDX:-207}"
DATASET="AMA-bench/AMA-bench"
SPLIT="test"
PREDICTOR="checkpoints/predictor_v2_best.pt"
RESULTS_DIR="benchmarks/results/ama_bench_episodes"
PROGRESS_FILE="$RESULTS_DIR/.progress"

mkdir -p "$RESULTS_DIR"
mkdir -p logs

# Load API key
if [ -f "$HOME/.jeval_secrets" ]; then
    source "$HOME/.jeval_secrets"
fi

if [ -z "${NVIDIA_API_KEY:-}" ]; then
    echo "ERROR: NVIDIA_API_KEY not set. Check ~/.jeval_secrets or .env" >&2
    exit 1
fi

export NVIDIA_API_KEY
export PYTHONPATH="$WORKDIR:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# Load conda/modules if on HPC
if command -v module &> /dev/null; then
    module purge 2>/dev/null || true
    module load miniconda3/24.11.1 2>/dev/null || true
fi

echo "========================================"
echo "AMA-Bench Sequential Runner (Option A)"
echo "========================================"
echo "Start: $(date)"
echo "Episodes: $START_IDX to $END_IDX"
echo "Results: $RESULTS_DIR"
echo "API key: ${NVIDIA_API_KEY:0:12}..."
echo ""

completed=0
skipped=0
failed=0

for idx in $(seq "$START_IDX" "$END_IDX"); do
    OUT="$RESULTS_DIR/episode_${idx}.json"
    
    # Skip if already complete (has 'score' field)
    if [ -f "$OUT" ] && python3 -c "
import json, sys
try:
    d = json.load(open('$OUT'))
    sys.exit(0 if 'score' in d else 1)
except:
    sys.exit(1)
" 2>/dev/null; then
        echo "[$(date +%H:%M:%S)] Episode $idx: already complete — skipping"
        ((skipped++)) || true
        continue
    fi
    
    echo "[$(date +%H:%M:%S)] Episode $idx: starting..."
    start_time=$(date +%s)
    
    # Run episode with timeout (5 minutes per episode)
    if timeout 300 python3 benchmarks/run_ama_episode.py \
        --episode-idx "$idx" \
        --dataset "$DATASET" \
        --split "$SPLIT" \
        --predictor "$PREDICTOR" \
        --out "$OUT" 2>&1 | tee -a "logs/ama_episode_${idx}.log"; then
        
        elapsed=$(($(date +%s) - start_time))
        ((completed++)) || true
        
        # Extract score if present
        if [ -f "$OUT" ]; then
            score=$(python3 -c "import json; print(json.load(open('$OUT')).get('score', 'N/A'))" 2>/dev/null || echo "N/A")
            echo "[$(date +%H:%M:%S)] Episode $idx: ✓ complete (${elapsed}s, score=$score)"
        else
            echo "[$(date +%H:%M:%S)] Episode $idx: ✓ done but no output (${elapsed}s)"
        fi
    else
        exit_code=$?
        elapsed=$(($(date +%s) - start_time))
        ((failed++)) || true
        
        if [ $exit_code -eq 124 ]; then
            echo "[$(date +%H:%M:%S)] Episode $idx: ✗ TIMEOUT (${elapsed}s)"
        else
            echo "[$(date +%H:%M:%S)] Episode $idx: ✗ FAILED exit=$exit_code (${elapsed}s)"
        fi
    fi
    
    # Save progress
    echo "$idx" > "$PROGRESS_FILE"
    
    # Small sleep to avoid hammering the login node
    sleep 2
done

echo ""
echo "========================================"
echo "Summary"
echo "========================================"
echo "Completed: $completed"
echo "Skipped:   $skipped"
echo "Failed:    $failed"
echo "End: $(date)"
