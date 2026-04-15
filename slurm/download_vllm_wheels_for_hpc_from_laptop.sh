#!/bin/bash
# Run on your LAPTOP (macOS or Linux) to fetch **Linux** wheels for Northeastern Explorer,
# then copy the directory to HPC and run: sbatch slurm/install_vllm_offline.sh
#
#   mkdir -p ~/jeval-wheels && cd ~/jeval-wheels
#   bash /path/to/Jeval/slurm/download_vllm_wheels_for_hpc_from_laptop.sh
#   rsync -avz --progress -e "ssh -i ~/.ssh/id_ed25519_explorer_shyam" ./jeval-pip-wheels-linux/ \
#     shyamsundar.p@login.explorer.northeastern.edu:~/scratch/jeval-pip-wheels/
#
# Requires: Python 3 with pip (any version on the laptop — we pass --python-version 310).
# If this fails to resolve vllm deps, use a Linux x86_64 environment (Docker) and run
# slurm/download_vllm_wheels.sh there instead, then rsync the folder.
#
# Env:
#   JEVAL_LAPTOP_WHEEL_DIR — output dir (default: ./jeval-pip-wheels-linux)
#   JEVAL_TARGET_PY        — minor version on Explorer venv (default: 310 = 3.10)

set -euo pipefail

WHEEL_DIR="${JEVAL_LAPTOP_WHEEL_DIR:-./jeval-pip-wheels-linux}"
TARGET_PY="${JEVAL_TARGET_PY:-310}"
# manylinux2014 is widely accepted by pip on EL9; matches typical torch manylinux wheels.
PLATFORM="${JEVAL_TARGET_PLATFORM:-manylinux2014_x86_64}"
TORCH_CUDA_INDEX="${TORCH_CUDA_INDEX:-https://download.pytorch.org/whl/cu121}"
PYPI="https://pypi.org/simple"

# pip cross-download ABI tag for CPython 3.10
PY_TAG=(--python-version "$TARGET_PY" --implementation cp --abi "cp${TARGET_PY}")

mkdir -p "$WHEEL_DIR"

dl() {
  python3 -m pip download -d "$WHEEL_DIR" "$@" \
    --platform "$PLATFORM" \
    "${PY_TAG[@]}" \
    --only-binary=:all:
}

echo "=== Linux wheels for HPC (from laptop) ==="
echo "WHEEL_DIR=$WHEEL_DIR  platform=$PLATFORM  cpython_tag=cp${TARGET_PY}"
echo "If Explorer uses Python 3.11 venv, set JEVAL_TARGET_PY=311 (etc.)."
echo ""

echo "--- pip / setuptools / wheel ---"
dl "pip>=24.0" "setuptools>=69.0" "wheel>=0.43.0" --extra-index-url "$PYPI"

echo "--- torch + torchvision (cu121 index first — ~800MB wheel is CUDA) ---"
python3 -m pip download -d "$WHEEL_DIR" "torch==2.4.0" "torchvision==0.19.0" \
  --platform "$PLATFORM" \
  "${PY_TAG[@]}" \
  --only-binary=:all: \
  --index-url "$TORCH_CUDA_INDEX" \
  --extra-index-url "$PYPI"

echo "--- vllm (wheel only; macOS pip cannot resolve Linux-only sdists for deps e.g. pyairports) ---"
python3 -m pip download -d "$WHEEL_DIR" "vllm==0.6.3" \
  --platform "$PLATFORM" \
  "${PY_TAG[@]}" \
  --only-binary=:all: \
  --no-deps \
  --index-url "$TORCH_CUDA_INDEX" \
  --extra-index-url "$PYPI" \
  --find-links "$WHEEL_DIR"

if [ -n "${JEVAL_WHEEL_EXTRA_PKGS:-}" ]; then
  echo "--- extras: $JEVAL_WHEEL_EXTRA_PKGS ---"
  python3 -m pip download -d "$WHEEL_DIR" $JEVAL_WHEEL_EXTRA_PKGS \
    --platform "$PLATFORM" \
    "${PY_TAG[@]}" \
    --index-url "$TORCH_CUDA_INDEX" \
    --extra-index-url "$PYPI" \
    --find-links "$WHEEL_DIR"
fi

echo ""
echo "=== Laptop stage DONE ==="
echo "1) rsync this folder to Explorer scratch (see header)."
echo "2) On Explorer LOGIN:  bash slurm/merge_vllm_wheel_deps_on_login.sh"
echo "3) sbatch slurm/install_vllm_offline.sh"
