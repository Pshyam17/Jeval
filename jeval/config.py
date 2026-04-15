"""
Jeval Configuration

Centralized configuration for models, endpoints, and key parameters.
Override via environment variables or by importing and modifying before use.
"""
import os
from pathlib import Path

# ============================================================================
# Model Names & Endpoints
# ============================================================================

# Sentence encoder (frozen, for embeddings)
ENCODER_MODEL = "sentence-transformers/all-mpnet-base-v2"
ENCODER_DIM = 768

# Content classifier (NLI cross-encoder)
CLASSIFIER_MODEL_FAST = "cross-encoder/nli-MiniLM2-L6-H768"  # Default: fast
CLASSIFIER_MODEL_PROD = "cross-encoder/nli-deberta-v3-large"  # Prod: accurate but slow

# LLM for compression and judging (NIM or local)
# NIM (API-based)
NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
NIM_MODEL = "mistralai/mistral-small-4-119b-2603"  # Updated 2026-04-15
NIM_API_KEY_ENV = "NVIDIA_API_KEY"

# Local inference (HPC cached model)
# Set LOCAL_MODEL_PATH to use local inference instead of NIM API
LOCAL_MODEL_PATH = os.getenv(
    "JEVAL_LOCAL_MODEL_PATH",
    # Default to HPC cached Mistral-Small-3.1 if present
    str(Path.home() / ".cache/huggingface/models--mistralai--Mistral-Small-3.1-24B-Instruct-2503/snapshots/68faf511d618ef198fef186659617cfd2eb8e33a")
    if (Path.home() / ".cache/huggingface/models--mistralai--Mistral-Small-3.1-24B-Instruct-2503/snapshots/68faf511d618ef198fef186659617cfd2eb8e33a").exists()
    else None
)

# Inference mode: "nim" (API) or "local" (cached model)
INFERENCE_MODE = os.getenv("JEVAL_INFERENCE_MODE", "nim")  # "nim" or "local"

# Local AMA-bench runner (`benchmarks/run_ama_episode_local.py`) backend
# "vllm" — recommended for Mistral-Small-3.1 / mistral3 (see vLLM docs)
# "hf"   — Hugging Face `transformers` generate() (may fail on mistral3 + older transformers)
LOCAL_LLM_BACKEND = os.getenv("JEVAL_LOCAL_LLM_BACKEND", "vllm").strip().lower()

# HF repo id used for tokenizer hub resolution when needed (weights still loaded from --model-path)
LOCAL_MODEL_REPO_ID = os.getenv(
    "JEVAL_LOCAL_MODEL_REPO_ID",
    "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
)

# vLLM weight/tokenizer layout: "hf" matches a standard HF snapshot cache; "mistral" uses
# tokenizer_mode/config_format/load_format=mistral (see vLLM examples/offline_inference/mistral-small.py)
VLLM_MODEL_FORMAT = os.getenv("JEVAL_VLLM_MODEL_FORMAT", "hf").strip().lower()

VLLM_MAX_MODEL_LEN = int(os.getenv("JEVAL_VLLM_MAX_MODEL_LEN", "16384"))
VLLM_GPU_MEMORY_UTILIZATION = float(os.getenv("JEVAL_VLLM_GPU_MEMORY_UTILIZATION", "0.90"))
VLLM_TENSOR_PARALLEL_SIZE = int(os.getenv("JEVAL_VLLM_TENSOR_PARALLEL_SIZE", "1"))
VLLM_ENFORCE_EAGER = os.getenv("JEVAL_VLLM_ENFORCE_EAGER", "0") == "1"

# ============================================================================
# Predictor Checkpoints
# ============================================================================

# Default predictor checkpoint (relative to repo root)
PREDICTOR_DEFAULT = "checkpoints/predictor_v2_best.pt"

# Alternative checkpoints
PREDICTOR_V3_LR3E4 = "checkpoints/predictor_v3_lr3e-04.pt"
PREDICTOR_V3_LR1E4 = "checkpoints/predictor_v3_lr1e-04.pt"

# ============================================================================
# Datasets
# ============================================================================

# AMA-Bench
AMA_BENCH_DATASET = "AMA-bench/AMA-bench"
AMA_BENCH_SPLIT = "test"
AMA_BENCH_TOTAL_EPISODES = 208

# SWE-Bench
SWE_BENCH_DATASET = "princeton-nlp/SWE-bench"
SWE_BENCH_SPLIT = "test"

# ============================================================================
# Memory System Parameters
# ============================================================================

# EPE thresholds
ALPHA_DEFAULT = 0.5  # Cosine vs schema gap blend (CombinedEPE)
BETA_DEFAULT = 1.0   # Budget modulation strength
HIGH_CONFIDENCE = 0.7  # Hot cache routing threshold
LOW_CONFIDENCE = 0.4   # Cold storage routing threshold

# Self-healing (miss-triggered recompression)
MISS_THRESHOLD = 2        # Trigger rewrite after N misses
MIN_REWRITE_GAP = 5       # Cooldown turns between rewrites
EVICTION_WEIGHTS = (0.3, 0.4, 0.2, 0.1)  # time, miss, hit, novelty

# Novelty gate
NOVELTY_WORKING_SET_SIZE = 50  # Recent embeddings for novelty check

# Hot cache limits
HOT_CACHE_MAX_TOKENS = 8000

# ============================================================================
# Compression Parameters
# ============================================================================

# LLM compression timeouts
COMPRESSION_TIMEOUT_S = 3.0  # Per-segment timeout
COMPRESSION_MAX_RETRIES = 2  # Fidelity gate retries before extractive fallback

# Budget floors per content type
BUDGET_FLOOR = {
    "FACTUAL": 0.75,
    "CONTRASTIVE": 0.55,
    "CAUSAL": 0.70,
    "ENTITY": 0.70,
    "TEMPORAL": 0.55,
    "BACKGROUND": 0.20,
}

# Budget sensitivity per content type
BUDGET_SENSITIVITY = {
    "FACTUAL": 0.30,
    "CONTRASTIVE": 0.25,
    "CAUSAL": 0.28,
    "ENTITY": 0.27,
    "TEMPORAL": 0.22,
    "BACKGROUND": 0.15,
}

# Risk weights for EPE decomposition
RISK_WEIGHTS = {
    "FACTUAL": 1.0,
    "CONTRASTIVE": 0.95,
    "CAUSAL": 0.90,
    "ENTITY": 0.85,
    "TEMPORAL": 0.75,
    "BACKGROUND": 0.50,
}

# ============================================================================
# Training Parameters
# ============================================================================

TRAIN_EPOCHS = 100
TRAIN_BATCH_SIZE = 32
TRAIN_LR_DEFAULT = 3e-4
TRAIN_WARMUP_RATIO = 0.1

# Target EPE separation (faithful vs lossy)
TARGET_EPE_SEPARATION = 5.0

# ============================================================================
# Paths
# ============================================================================

# Default database path
DEFAULT_DB_PATH = ".jeval/memory.db"

# Logs directory
LOGS_DIR = "logs"

# Results directory
RESULTS_DIR = "benchmarks/results"

# ============================================================================
# Environment Flags
# ============================================================================

# Offline mode (use cached datasets/models only)
TRANSFORMERS_OFFLINE = os.getenv("TRANSFORMERS_OFFLINE", "1") == "1"
HF_DATASETS_OFFLINE = os.getenv("HF_DATASETS_OFFLINE", "1") == "1"
TOKENIZERS_PARALLELISM = os.getenv("TOKENIZERS_PARALLELISM", "false") == "true"

# ============================================================================
# Utility Functions
# ============================================================================

def get_nim_client():
    """Get OpenAI client for NIM endpoint."""
    from openai import OpenAI
    return OpenAI(
        api_key=os.environ[NIM_API_KEY_ENV],
        base_url=NIM_BASE_URL,
    )


def get_local_model_path():
    """Get local model path if configured and exists."""
    if LOCAL_MODEL_PATH and Path(LOCAL_MODEL_PATH).exists():
        return LOCAL_MODEL_PATH
    return None


def use_local_inference():
    """Check if local inference should be used."""
    return INFERENCE_MODE == "local" and get_local_model_path() is not None


def summary():
    """Print current configuration."""
    print("=" * 60)
    print("Jeval Configuration")
    print("=" * 60)
    print(f"Encoder:           {ENCODER_MODEL}")
    print(f"Classifier:        {CLASSIFIER_MODEL_FAST}")
    print(f"Inference mode:    {INFERENCE_MODE}")
    if INFERENCE_MODE == "nim":
        print(f"  NIM endpoint:    {NIM_BASE_URL}")
        print(f"  NIM model:       {NIM_MODEL}")
        print(f"  API key set:     {bool(os.getenv(NIM_API_KEY_ENV))}")
    else:
        print(f"  Local model:     {LOCAL_MODEL_PATH}")
        print(f"  Model exists:    {Path(LOCAL_MODEL_PATH).exists() if LOCAL_MODEL_PATH else False}")
        print(f"  Local LLM:       {LOCAL_LLM_BACKEND} (vLLM format={VLLM_MODEL_FORMAT})")
    print(f"Predictor:         {PREDICTOR_DEFAULT}")
    print(f"AMA-Bench:         {AMA_BENCH_DATASET} ({AMA_BENCH_TOTAL_EPISODES} episodes)")
    print(f"Alpha (EPE blend): {ALPHA_DEFAULT}")
    print(f"Offline mode:      Transformers={TRANSFORMERS_OFFLINE}, Datasets={HF_DATASETS_OFFLINE}")
    print("=" * 60)


if __name__ == "__main__":
    summary()
