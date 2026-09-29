#!/bin/bash
#SBATCH --job-name=jeval-epe
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=jeval-epe-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
: "${JEVAL_PYTHON:?Set JEVAL_PYTHON to your venv's python executable}"
: "${JEVAL_EMBEDDINGS:?Set JEVAL_EMBEDDINGS to the input .npz path}"
: "${JEVAL_CHECKPOINT:?Set JEVAL_CHECKPOINT to the output .pt path}"
"${JEVAL_PYTHON}" -m jeval_v2.experiment train --embeddings "${JEVAL_EMBEDDINGS}" --output "${JEVAL_CHECKPOINT}" --device cuda --seed "${JEVAL_SEED:-0}"
