#!/bin/bash
#SBATCH --job-name=jeval-compress
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=jeval-compress-%j.log
set -euo pipefail
# the venv python needs libpython from this module on compute nodes
module load "${JEVAL_PYTHON_MODULE:-python/3.13.5}"
: "${JEVAL_REPO_ROOT:?Set JEVAL_REPO_ROOT to the absolute repository root}"
cd "${JEVAL_REPO_ROOT}"
: "${JEVAL_PYTHON:?Set JEVAL_PYTHON to your venv Python executable}"
: "${JEVAL_LOCOMO:?Set JEVAL_LOCOMO to the official locomo10.json path}"
: "${JEVAL_RUN_DIR:?Set JEVAL_RUN_DIR to the output directory for this run}"
: "${JEVAL_LLM:?Set JEVAL_LLM to the compressor/reader model}"
: "${JEVAL_LLM_REVISION:?Set JEVAL_LLM_REVISION to the exact model commit}"
: "${JEVAL_RATIO:?Set JEVAL_RATIO to the target compressed/original word ratio}"
: "${JEVAL_CACHE_DIR:?Set JEVAL_CACHE_DIR to an absolute scratch/project cache path}"
export HF_HOME="${JEVAL_CACHE_DIR}/hf"
export HF_HUB_CACHE="${HF_HOME}/hub"
# compute nodes have no route to huggingface.co; models are pre-downloaded on the login node
export HF_HUB_OFFLINE=1
mkdir -p "${HF_HUB_CACHE}" "${JEVAL_RUN_DIR}"
"${JEVAL_PYTHON}" -m jeval_v2.locomo segment --input "${JEVAL_LOCOMO}" --output "${JEVAL_RUN_DIR}/segments.jsonl" --turns "${JEVAL_TURNS:-6}"
"${JEVAL_PYTHON}" -m jeval_v2.generate compress --segments "${JEVAL_RUN_DIR}/segments.jsonl" --output "${JEVAL_RUN_DIR}/candidates.jsonl" --model "${JEVAL_LLM}" --revision "${JEVAL_LLM_REVISION}" --ratio "${JEVAL_RATIO}" --n "${JEVAL_N:-8}" --seed "${JEVAL_SEED:-0}"
