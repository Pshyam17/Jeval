#!/usr/bin/env python3
"""
benchmarks/run_ama_episode_local.py

Option D: Local inference variant of run_ama_episode.py
Uses a cached Mistral model on GPU compute nodes instead of the NIM API.

Default backend is vLLM (`JEVAL_LOCAL_LLM_BACKEND=vllm`), which matches upstream
support for Mistral-Small-3.1. Optional `--backend hf` uses `transformers`+`generate`
(may fail for mistral3 checkpoints depending on your transformers build).

Runs one AMA-Bench episode end-to-end:
  load episode → compress trajectory → store in JevalMemory →
  retrieve+answer per QA pair → judge → write result JSON

Designed for SLURM GPU nodes with cached models (no internet).

CLI:
    python benchmarks/run_ama_episode_local.py \
      --episode-idx 0 \
      --dataset AMA-bench/AMA-bench \
      --split test \
      --predictor checkpoints/predictor_v2_best.pt \
      --model-path ~/.cache/huggingface/.../snapshots/<hash> \
      --backend vllm \
      --vllm-format hf \
      --out benchmarks/results/ama_bench_episodes/episode_0.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

import torch

from jeval import config
from jeval.benchmarks.ama_bench_eval import (
    ANSWER_PROMPT,
    JUDGE_PROMPT,
    JevalMemory,
)
from jeval.compress.adaptive import AdaptiveCompressor
from jeval.compress.extractive import ExtractiveBackend


# ── Local LLM backends (vLLM preferred, HF optional) ─────────────────────────


class LocalCompletionBackend(Protocol):
    """Minimal interface for answer/judge completions."""

    def complete(self, prompt: str, max_tokens: int) -> str: ...


class HFTransformersBackend:
    """HF `transformers` + `generate()` (legacy / debugging)."""

    def __init__(self, model_path: str) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        repo_id = config.LOCAL_MODEL_REPO_ID
        cache_dir = str(Path.home() / ".cache/huggingface")
        resolved = str(Path(model_path).resolve())

        print(
            f"Loading Mistral via HF backend from repo_id={repo_id} "
            f"(cache_dir={cache_dir}, snapshot={resolved})...",
            flush=True,
        )
        start = time.time()
        self._tokenizer = AutoTokenizer.from_pretrained(
            repo_id,
            cache_dir=cache_dir,
            fix_mistral_regex=True,
            use_fast=False,
            local_files_only=True,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            repo_id,
            cache_dir=cache_dir,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True,
        )
        self._model.eval()
        print(f"HF backend ready in {time.time() - start:.1f}s", flush=True)

    def complete(self, prompt: str, max_tokens: int = 512) -> str:
        messages = [{"role": "user", "content": prompt}]
        inputs = self._tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(self._model.device)

        with torch.no_grad():
            outputs = self._model.generate(
                inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.1,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        generated = outputs[0][inputs.shape[1] :]
        return self._tokenizer.decode(generated, skip_special_tokens=True).strip()


class VLLMTextBackend:
    """
    vLLM offline engine. Uses `llm.chat()` with plain string user content (text-only).

    Weight layout is controlled by `weight_format`: "hf" (default) matches a normal HF
    snapshot directory; "mistral" sets tokenizer/config/load format to mistral per vLLM
    upstream examples (see vLLM `examples/offline_inference/mistral-small.py`).
    """

    def __init__(self, model_path: str, weight_format: str) -> None:
        try:
            from vllm import LLM
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "vLLM backend requested but `vllm` is not installed. "
                "On GPU nodes: `pip install 'vllm>=0.6'` (match your CUDA/torch stack; "
                "Mistral-Small-3.1 may need a recent vLLM). "
                "Or run with `--backend hf` if your transformers build supports this checkpoint."
            ) from exc

        try:
            from vllm.sampling_params import SamplingParams
        except ImportError:  # pragma: no cover - older vLLM layouts
            from vllm import SamplingParams  # type: ignore

        self._SamplingParams = SamplingParams
        weight_format = weight_format.lower().strip()
        if weight_format not in ("hf", "mistral"):
            raise ValueError(f"weight_format must be 'hf' or 'mistral', got {weight_format!r}")

        model = str(Path(model_path).resolve())
        print(
            f"Loading Mistral via vLLM (model={model}, format={weight_format})...",
            flush=True,
        )
        start = time.time()

        kwargs: Dict[str, Any] = dict(
            model=model,
            trust_remote_code=True,
            dtype="auto",
            max_model_len=config.VLLM_MAX_MODEL_LEN,
            tensor_parallel_size=config.VLLM_TENSOR_PARALLEL_SIZE,
            gpu_memory_utilization=config.VLLM_GPU_MEMORY_UTILIZATION,
            enforce_eager=config.VLLM_ENFORCE_EAGER,
        )
        if weight_format == "mistral":
            kwargs["tokenizer_mode"] = "mistral"
            kwargs["config_format"] = "mistral"
            kwargs["load_format"] = "mistral"

        self._llm = LLM(**kwargs)
        print(f"vLLM backend ready in {time.time() - start:.1f}s", flush=True)

    def complete(self, prompt: str, max_tokens: int = 512) -> str:
        messages = [{"role": "user", "content": prompt}]
        sp = self._SamplingParams(max_tokens=max_tokens, temperature=0.1)
        outputs = self._llm.chat(messages, sampling_params=sp)
        return outputs[0].outputs[0].text.strip()


def create_local_backend(
    name: str,
    model_path: str,
    *,
    vllm_format: Optional[str] = None,
) -> LocalCompletionBackend:
    key = name.strip().lower()
    if key == "vllm":
        fmt = (vllm_format or config.VLLM_MODEL_FORMAT).lower().strip()
        return VLLMTextBackend(model_path, fmt)
    if key == "hf":
        return HFTransformersBackend(model_path)
    raise ValueError(f"Unknown local backend {name!r} (expected 'vllm' or 'hf')")


# ── Scoring ───────────────────────────────────────────────────────────────────

def _answer(llm: LocalCompletionBackend, mem: JevalMemory, question: str, task: str) -> str:
    context = mem.memory_retrieve(question, top_k=5)
    if not context:
        context = mem.full_memory[:3000]
    prompt = ANSWER_PROMPT.format(context=context, task=task[:300], question=question)
    return llm.complete(prompt, max_tokens=300)


def _judge(llm: LocalCompletionBackend, question: str, reference: str, predicted: str) -> tuple[bool, str]:
    prompt = JUDGE_PROMPT.format(
        question=question,
        reference=reference,
        predicted=predicted,
    )
    raw = llm.complete(prompt, max_tokens=150)
    try:
        clean = raw.strip().strip("```json").strip("```").strip()
        result = json.loads(clean)
        return bool(result["correct"]), result.get("reasoning", "")
    except Exception:
        # Fallback: keyword matching
        correct = any(w in raw.lower() for w in ["true", "correct", "yes"])
        return correct, raw[:100]


# ── Episode loader ────────────────────────────────────────────────────────────

def _load_episode(dataset_name: str, split: str, idx: int) -> Dict[str, Any]:
    from datasets import load_dataset  # type: ignore

    ds = load_dataset(dataset_name, split=split)
    if idx >= len(ds):
        raise IndexError(f"episode_idx {idx} >= dataset size {len(ds)}")
    return ds[idx]


# ── Predictor loading ─────────────────────────────────────────────────────────

def _load_predictor(ckpt: Optional[str]):
    """Load predictor checkpoint if available."""
    if not ckpt or not Path(ckpt).exists():
        return None
    from jeval.encoders.predictor_head import PreLNTransformerPredictor
    from jeval.encoders.sentence_encoder import FrozenEncoder

    enc = FrozenEncoder()
    pred = PreLNTransformerPredictor(enc.dim())
    pred.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False))
    pred.eval()
    return pred


# ── Main episode run ──────────────────────────────────────────────────────────

def run_episode(
    episode_idx: int,
    dataset_name: str,
    split: str,
    predictor_path: Optional[str],
    model_path: str,
    out_path: Path,
    local_backend: str,
    vllm_format: Optional[str],
) -> None:
    """Run one AMA-Bench episode with local inference."""
    ep = _load_episode(dataset_name, split, episode_idx)

    ep_id = ep["episode_id"]
    task = ep.get("task", "")
    qas = ep["qa_pairs"] if isinstance(ep["qa_pairs"], list) else json.loads(ep["qa_pairs"])
    traj = ep["trajectory"] if isinstance(ep["trajectory"], list) else json.loads(ep["trajectory"])

    # Build trajectory text
    traj_lines: List[str] = []
    orig_tokens = 0
    for turn in traj:
        idx = turn.get("turn_idx", 0)
        act = turn.get("action", "")
        obs = turn.get("observation", "")
        if act:
            line = f"Step {idx} action: {act[:800]}"
            traj_lines.append(line)
            orig_tokens += len(line.split())
        if obs:
            line = f"Step {idx} observation: {obs[:800]}"
            traj_lines.append(line)
            orig_tokens += len(line.split())
    traj_text = "\n".join(traj_lines)

    # Build memory (compression uses extractive backend, not LLM)
    predictor = _load_predictor(predictor_path)
    mem = JevalMemory()
    if predictor is not None:
        mem.compressor = AdaptiveCompressor(
            predictor=predictor,
            backend=ExtractiveBackend(),
        )
    print(f"Compressing trajectory ({orig_tokens} tokens)...", flush=True)
    mem.memory_construction(traj_text, task=task)

    compressed_tokens = len(mem.full_memory.split())
    compression_ratio = compressed_tokens / max(orig_tokens, 1)
    print(f"Compression complete: {orig_tokens} → {compressed_tokens} tokens (ratio={compression_ratio:.2f})", flush=True)

    # Load LLM after compression so GPU stays free during embedding-heavy compression.
    llm_backend = create_local_backend(
        local_backend,
        model_path,
        vllm_format=vllm_format,
    )

    per_question: List[dict] = []
    scores: List[float] = []

    print(f"Processing {len(qas)} QA pairs...", flush=True)
    for i, qa in enumerate(qas):
        question = qa["question"]
        reference = qa["answer"]
        qa_type = qa.get("type", "")
        q_uuid = qa.get("question_uuid", "")

        print(f"  Q{i+1}/{len(qas)}: answering...", flush=True)
        answer = _answer(llm_backend, mem, question, task)
        print(f"  Q{i+1}/{len(qas)}: judging...", flush=True)
        correct, reason = _judge(llm_backend, question, reference, answer)
        score = 1.0 if correct else 0.0
        scores.append(score)
        print(f"  Q{i+1}/{len(qas)}: score={score} ({'✓' if correct else '✗'})", flush=True)

        per_question.append({
            "question_uuid": q_uuid,
            "type": qa_type,
            "question": question,
            "answer": answer,
            "score": score,
            "reasoning": reason,
        })

    result = {
        "episode_idx": episode_idx,
        "episode_id": ep_id,
        "score": float(sum(scores) / max(len(scores), 1)),
        "n_questions": len(qas),
        "correct": int(sum(scores)),
        "total": len(scores),
        "per_question": per_question,
        "compression_ratio": compression_ratio,
        "original_tokens": orig_tokens,
        "compressed_tokens": compressed_tokens,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(
        f"episode {episode_idx}  score={result['score']:.2f}  "
        f"correct={int(sum(scores))}/{len(qas)}  ratio={compression_ratio:.2f}  → {out_path}",
        flush=True,
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run one AMA-Bench episode (local inference)")
    parser.add_argument("--episode-idx", type=int, required=True, help="Episode index (0-207)")
    parser.add_argument("--dataset", type=str, default=config.AMA_BENCH_DATASET, help="HF dataset name")
    parser.add_argument("--split", type=str, default=config.AMA_BENCH_SPLIT, help="Dataset split")
    parser.add_argument("--predictor", type=str, default=config.PREDICTOR_DEFAULT, help="Predictor checkpoint path")
    parser.add_argument("--model-path", type=str, default=config.LOCAL_MODEL_PATH, help="Local Mistral model path")
    parser.add_argument(
        "--backend",
        type=str,
        choices=("vllm", "hf"),
        default=None,
        help="Local inference engine (default: JEVAL_LOCAL_LLM_BACKEND or jeval.config.LOCAL_LLM_BACKEND)",
    )
    parser.add_argument(
        "--vllm-format",
        type=str,
        choices=("hf", "mistral"),
        default=None,
        help="vLLM weight/tokenizer layout (default: JEVAL_VLLM_MODEL_FORMAT); ignored for --backend hf",
    )
    parser.add_argument("--out", type=str, required=True, help="Output JSON path")
    args = parser.parse_args()

    if not args.model_path or not Path(args.model_path).exists():
        print(f"ERROR: Local model not found at {args.model_path}", file=sys.stderr)
        print("Set --model-path to a valid cached model directory", file=sys.stderr)
        sys.exit(1)

    out_path = Path(args.out)
    backend_name = (args.backend or config.LOCAL_LLM_BACKEND).lower().strip()

    try:
        run_episode(
            episode_idx=args.episode_idx,
            dataset_name=args.dataset,
            split=args.split,
            predictor_path=args.predictor,
            model_path=args.model_path,
            out_path=out_path,
            local_backend=backend_name,
            vllm_format=args.vllm_format,
        )
    except Exception as exc:
        # Always exit 0; write error to JSON so aggregate step can distinguish
        error_result = {
            "episode_idx": args.episode_idx,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(error_result, indent=2))
        print(f"ERROR episode {args.episode_idx}: {exc}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
