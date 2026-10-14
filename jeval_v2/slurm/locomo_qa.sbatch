#!/bin/bash
#SBATCH --job-name=jeval-qa
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=jeval-qa-%j.log
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
: "${JEVAL_RATIO:?Set JEVAL_RATIO to the same ratio used for compress}"
: "${JEVAL_CACHE_DIR:?Set JEVAL_CACHE_DIR to an absolute scratch/project cache path}"
export HF_HOME="${JEVAL_CACHE_DIR}/hf"
export HF_HUB_CACHE="${HF_HOME}/hub"
mkdir -p "${HF_HUB_CACHE}"
"${JEVAL_PYTHON}" -m jeval_v2.locomo select --segments "${JEVAL_RUN_DIR}/segments.jsonl" --candidates "${JEVAL_RUN_DIR}/candidates.jsonl" --scores "${JEVAL_RUN_DIR}"/fold-*/scores.jsonl --output-dir "${JEVAL_RUN_DIR}/selected" --ratio "${JEVAL_RATIO}" --seed "${JEVAL_SEED:-0}"
"${JEVAL_PYTHON}" -m jeval_v2.generate qa --locomo "${JEVAL_LOCOMO}" --memories "${JEVAL_RUN_DIR}/selected/memories.jsonl" --output "${JEVAL_RUN_DIR}/predictions.jsonl" --model "${JEVAL_LLM}" --revision "${JEVAL_LLM_REVISION}" --seed "${JEVAL_SEED:-0}"
"${JEVAL_PYTHON}" -m jeval_v2.locomo report --predictions "${JEVAL_RUN_DIR}/predictions.jsonl" | tee "${JEVAL_RUN_DIR}/report.json"
