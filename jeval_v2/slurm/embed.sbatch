#!/bin/bash
#SBATCH --job-name=jeval-embed
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=jeval-embed-%j.log
set -euo pipefail
: "${JEVAL_REPO_ROOT:?Set JEVAL_REPO_ROOT to the absolute repository root}"
cd "${JEVAL_REPO_ROOT}"
: "${JEVAL_PYTHON:?Set JEVAL_PYTHON to your venv Python executable}"
: "${JEVAL_PAIRS:?Set JEVAL_PAIRS to the JSONL pair path}"
: "${JEVAL_EMBEDDINGS:?Set JEVAL_EMBEDDINGS to the output .npz path}"
: "${JEVAL_ENCODER:?Set JEVAL_ENCODER to a pinned encoder model/revision}"
: "${JEVAL_ENCODER_REVISION:?Set JEVAL_ENCODER_REVISION to the exact model commit}"
: "${JEVAL_CACHE_DIR:?Set JEVAL_CACHE_DIR to an absolute scratch/project cache path}"
export HF_HOME="${JEVAL_CACHE_DIR}/hf"
export HF_HUB_CACHE="${HF_HOME}/hub"
mkdir -p "${HF_HUB_CACHE}"
"${JEVAL_PYTHON}" -m jeval_v2.experiment embed --pairs "${JEVAL_PAIRS}" --output "${JEVAL_EMBEDDINGS}" --encoder "${JEVAL_ENCODER}" --encoder-revision "${JEVAL_ENCODER_REVISION}" --device cuda
