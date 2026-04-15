#!/bin/bash
# Test NVIDIA NIM API connectivity from HPC compute node
#
# Usage:
#   sbatch slurm/test_nim_api.sh
#
#SBATCH --job-name=jeval-nim-test
#SBATCH --account=cs6140.202630
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=logs/nim_test_%j.out
#SBATCH --error=logs/nim_test_%j.err

set -euo pipefail

WORKDIR="${SLURM_SUBMIT_DIR:-$HOME/jeval/Jeval-1}"
cd "$WORKDIR"
mkdir -p logs

module purge
module load miniconda3/24.11.1

# Load API key
if [ -f "$HOME/.jeval_secrets" ]; then
    source "$HOME/.jeval_secrets"
fi

if [ -z "${NVIDIA_API_KEY:-}" ]; then
    echo "ERROR: NVIDIA_API_KEY not set" >&2
    exit 1
fi

echo "=== NIM API test  node=$SLURMD_NODENAME  start=$(date) ==="
echo "Key length: ${#NVIDIA_API_KEY}"

# Test 1: curl the API endpoint
echo ""
echo "=== Test 1: curl ping ==="
curl -s --max-time 10 -w "\nHTTP_CODE: %{http_code}\n" \
  -H "Authorization: Bearer $NVIDIA_API_KEY" \
  https://integrate.api.nvidia.com/v1/models 2>&1 | head -50

# Test 2: Python openai client
echo ""
echo "=== Test 2: Python OpenAI client ==="
python -u << 'PYEOF'
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["NVIDIA_API_KEY"],
    base_url="https://integrate.api.nvidia.com/v1",
)

try:
    # List models
    print("Listing models...")
    models = client.models.list()
    print(f"Found {len(list(models))} models")
    
    # Try a minimal completion
    print("\nTesting completion...")
    resp = client.chat.completions.create(
        model="mistralai/mistral-small-3.1-24b-instruct-2503",
        messages=[{"role": "user", "content": "Reply with: OK"}],
        max_tokens=10,
        temperature=0.0,
    )
    answer = resp.choices[0].message.content.strip()
    print(f"Response: {answer}")
    print("\n✓ NIM API is accessible and working")
except Exception as exc:
    print(f"\n✗ NIM API failed: {exc}")
    import traceback
    traceback.print_exc()
PYEOF

echo ""
echo "=== done $(date) ==="
