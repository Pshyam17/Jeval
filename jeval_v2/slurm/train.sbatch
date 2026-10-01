#!/bin/bash
#SBATCH --job-name=jeval-epe
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --array=0-2
#SBATCH --output=jeval-epe-%A_%a.log
set -euo pipefail
: "${JEVAL_REPO_ROOT:?Set JEVAL_REPO_ROOT to the absolute repository root}"
cd "${JEVAL_REPO_ROOT}"
: "${JEVAL_PYTHON:?Set JEVAL_PYTHON to your venv Python executable}"
: "${JEVAL_EMBEDDINGS:?Set JEVAL_EMBEDDINGS to the input .npz path}"
: "${JEVAL_CHECKPOINT_DIR:?Set JEVAL_CHECKPOINT_DIR to the checkpoint directory}"
mkdir -p "${JEVAL_CHECKPOINT_DIR}"
seed="${SLURM_ARRAY_TASK_ID:?Submit as an array job}"
"${JEVAL_PYTHON}" -m jeval_v2.experiment train --embeddings "${JEVAL_EMBEDDINGS}" --output "${JEVAL_CHECKPOINT_DIR}/predictor-${seed}.pt" --device cpu --selection "${JEVAL_SELECTION:-mse}" --seed "${seed}"
