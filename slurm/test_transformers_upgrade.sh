#!/bin/bash
#SBATCH --job-name=test-transformers
#SBATCH --account=cs6140.202630
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=00:10:00
#SBATCH --output=logs/test_transformers_%j.out
#SBATCH --error=logs/test_transformers_%j.err

set -euo pipefail

echo "==================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start: $(date)"
echo "==================================="

# Load modules
module purge
module load miniconda3/24.11.1
module load cuda/12.3.0 2>/dev/null || module load cuda/12.3 2>/dev/null || true

# Set working directory
WORKDIR=~/jeval/Jeval-1
cd "$WORKDIR" || exit 1

# Load secrets
if [ -f ~/.jeval_secrets ]; then
    source ~/.jeval_secrets
fi

# Set Python path
export PYTHONPATH="${WORKDIR}:${PYTHONPATH:-}"

echo ""
echo "=== Verifying package versions ==="
python3 << 'EOF'
import sentence_transformers
import transformers
import torch

print(f"sentence-transformers: {sentence_transformers.__version__}")
print(f"transformers:          {transformers.__version__}")
print(f"torch:                 {torch.__version__}")
print(f"CUDA available:        {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device:           {torch.cuda.get_device_name(0)}")
EOF

echo ""
echo "=== Testing Mistral model loading ==="
python3 << 'EOF'
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from pathlib import Path
import torch

repo_id = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
cache_dir = str(Path.home() / ".cache/huggingface")

print(f"\n1. Loading config from {repo_id}...")
config = AutoConfig.from_pretrained(
    repo_id,
    cache_dir=cache_dir,
    local_files_only=True,
    trust_remote_code=True,
)
print(f"   ✓ Config type: {type(config).__name__}")

print(f"\n2. Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    repo_id,
    cache_dir=cache_dir,
    use_fast=False,
    local_files_only=True,
)
print(f"   ✓ Tokenizer type: {type(tokenizer).__name__}")
print(f"   ✓ Vocab size: {len(tokenizer)}")

print(f"\n3. Checking AutoModelForCausalLM compatibility...")
from transformers import AutoModelForCausalLM
model_class = AutoModelForCausalLM._model_mapping.get(type(config), None)
if model_class:
    print(f"   ✓ Config mapping found: {type(config).__name__} -> {model_class}")
else:
    print(f"   ✗ Config mapping NOT found for {type(config).__name__}")
    print(f"   Available Mistral configs:")
    for cfg_type in AutoModelForCausalLM._model_mapping.keys():
        if "mistral" in cfg_type.__name__.lower():
            print(f"     - {cfg_type.__name__}")

print(f"\n4. Attempting to load model (FP16, device_map=auto)...")
try:
    model = AutoModelForCausalLM.from_pretrained(
        repo_id,
        cache_dir=cache_dir,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
    )
    print(f"   ✓ Model loaded successfully!")
    print(f"   ✓ Model type: {type(model).__name__}")
    print(f"   ✓ Device: {model.device}")
    
    # Test a simple generation
    print(f"\n5. Testing generation...")
    messages = [{"role": "user", "content": "Hello"}]
    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)
    inputs = inputs.to(model.device)
    
    outputs = model.generate(inputs, max_new_tokens=10, do_sample=False)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"   ✓ Generation test passed")
    print(f"   Response snippet: {response[:100]}...")
    
except Exception as e:
    print(f"   ✗ Model loading failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n=== SUCCESS: All tests passed ===")
EOF

echo ""
echo "==================================="
echo "End: $(date)"
echo "==================================="
