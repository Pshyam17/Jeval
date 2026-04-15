#!/bin/bash
# Download wheels with plenty of RAM — use if login-node `download_vllm_wheels.sh` gets OOM-killed.
# Needs outbound HTTPS on the node (often true for `short`, not for GPU).
#
#   sbatch slurm/download_vllm_wheels_job.sh
#
#SBATCH --job-name=dl-vllm-whl
#SBATCH --account=cs6140.202630
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=logs/download_vllm_wheels_%j.out
#SBATCH --error=logs/download_vllm_wheels_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$HOME/jeval/Jeval-1}"
mkdir -p logs

module purge
module load miniconda3/24.11.1

bash slurm/download_vllm_wheels.sh
