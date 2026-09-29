#!/bin/bash
#SBATCH --job-name=jeval-embed
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=jeval-embed-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
: "${JEVAL_PYTHON:?Set JEVAL_PYTHON to your venv's python executable}"
: "${JEVAL_PAIRS:?Set JEVAL_PAIRS to the JSONL pair path}"
: "${JEVAL_EMBEDDINGS:?Set JEVAL_EMBEDDINGS to the output .npz path}"
"${JEVAL_PYTHON}" -m jeval_v2.experiment embed --pairs "${JEVAL_PAIRS}" --output "${JEVAL_EMBEDDINGS}" --device cuda
