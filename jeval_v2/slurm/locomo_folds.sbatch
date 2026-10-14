#!/bin/bash
#SBATCH --job-name=jeval-folds
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --array=0-4
#SBATCH --output=jeval-folds-%A_%a.log
set -euo pipefail
# the venv python needs libpython from this module on compute nodes
module load "${JEVAL_PYTHON_MODULE:-python/3.13.5}"
: "${JEVAL_REPO_ROOT:?Set JEVAL_REPO_ROOT to the absolute repository root}"
cd "${JEVAL_REPO_ROOT}"
: "${JEVAL_PYTHON:?Set JEVAL_PYTHON to your venv Python executable}"
: "${JEVAL_RUN_DIR:?Set JEVAL_RUN_DIR to the output directory for this run}"
: "${JEVAL_ENCODER:?Set JEVAL_ENCODER to a pinned encoder model}"
: "${JEVAL_ENCODER_REVISION:?Set JEVAL_ENCODER_REVISION to the exact encoder commit}"
: "${JEVAL_CACHE_DIR:?Set JEVAL_CACHE_DIR to an absolute scratch/project cache path}"
export HF_HOME="${JEVAL_CACHE_DIR}/hf"
export HF_HUB_CACHE="${HF_HOME}/hub"
mkdir -p "${HF_HUB_CACHE}"
fold="${SLURM_ARRAY_TASK_ID:?Submit as an array job}"
dir="${JEVAL_RUN_DIR}/fold-${fold}"
mkdir -p "${dir}"
# Array size 0-4 matches --folds 5: each conversation is tested exactly once.
"${JEVAL_PYTHON}" -m jeval_v2.locomo make-pairs --segments "${JEVAL_RUN_DIR}/segments.jsonl" --candidates "${JEVAL_RUN_DIR}/candidates.jsonl" --folds 5 --fold "${fold}" --output "${dir}/pairs.jsonl"
"${JEVAL_PYTHON}" -m jeval_v2.experiment embed --pairs "${dir}/pairs.jsonl" --output "${dir}/pairs.npz" --encoder "${JEVAL_ENCODER}" --encoder-revision "${JEVAL_ENCODER_REVISION}" --device cuda
"${JEVAL_PYTHON}" -m jeval_v2.experiment train --embeddings "${dir}/pairs.npz" --output "${dir}/predictor.pt" --device cuda --seed "${JEVAL_SEED:-0}"
"${JEVAL_PYTHON}" -m jeval_v2.experiment evaluate --embeddings "${dir}/pairs.npz" --checkpoint "${dir}/predictor.pt" --split test --output "${dir}/scores.jsonl" --device cuda --seed "${JEVAL_SEED:-0}"
