#!/bin/bash
# Same as merge_vllm_wheel_deps_on_login.sh but on Slurm `short` (avoids login OOM).
#
#   sbatch slurm/merge_vllm_wheel_deps_job.sh
#
#SBATCH --job-name=merge-vllm-whl
#SBATCH --account=cs6140.202630
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=logs/merge_vllm_wheel_deps_%j.out
#SBATCH --error=logs/merge_vllm_wheel_deps_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$HOME/jeval/Jeval-1}"
mkdir -p logs

module purge
module load miniconda3/24.11.1

export JEVAL_WHEEL_DIR="${JEVAL_WHEEL_DIR:-${SCRATCH:-$HOME/scratch}/jeval-pip-wheels}"
bash slurm/merge_vllm_wheel_deps_on_login.sh
