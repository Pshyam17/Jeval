#!/bin/bash
# Submit compress -> 5 fold jobs -> select/QA/report as a dependency chain.
set -euo pipefail
: "${JEVAL_ACCOUNT:?Set JEVAL_ACCOUNT}"
: "${JEVAL_GPU_PARTITION:?Set JEVAL_GPU_PARTITION}"
here="$(cd "$(dirname "$0")" && pwd)"
opts=(--parsable --export=ALL --account="${JEVAL_ACCOUNT}" --partition="${JEVAL_GPU_PARTITION}")
compress=$(sbatch "${opts[@]}" "${here}/locomo_compress.sbatch")
folds=$(sbatch "${opts[@]}" --dependency=afterok:"${compress}" "${here}/locomo_folds.sbatch")
qa=$(sbatch "${opts[@]}" --dependency=afterok:"${folds}" "${here}/locomo_qa.sbatch")
echo "compress=${compress} folds=${folds} qa=${qa}"
