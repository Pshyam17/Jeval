#!/usr/bin/env python3
"""
benchmarks/run_ama_episode.py

Runs one AMA-Bench episode end-to-end:
  load episode → compress trajectory → store in JevalMemory →
  retrieve+answer per QA pair → judge → write result JSON

Designed to run as a SLURM array task (one process per episode).
Exits with code 0 always; errors are serialized to the output JSON so the
aggregate step can distinguish real failures from bad scores.

CLI:
    python benchmarks/run_ama_episode.py \
      --episode-idx 0 \
      --dataset AMA-bench/AMA-bench \
      --split test \
      --predictor checkpoints/predictor_v2_best.pt \
      --out benchmarks/results/ama_bench_episodes/episode_0.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI

from jeval.benchmarks.ama_bench_eval import (
    ANSWER_PROMPT,
    JUDGE_PROMPT,
    JevalMemory,
    trajectory_to_session,
)
from jeval.compress.adaptive import AdaptiveCompressor
from jeval.compress.extractive import ExtractiveBackend

_MODEL = "mistralai/mistral-small-3.1-24b-instruct-2503"


# ── NIM client with retry ─────────────────────────────────────────────────────

def _client() -> OpenAI:
    return OpenAI(
        api_key=os.environ["NVIDIA_API_KEY"],
        base_url="https://integrate.api.nvidia.com/v1",
    )


def _complete(client: OpenAI, prompt: str, max_tokens: int = 512) -> str:
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.1,
            )
            return resp.choices[0].message.content.strip()
        except Exception as exc:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)
    return ""  # unreachable but satisfies type checker


# ── Scoring ───────────────────────────────────────────────────────────────────

def _answer(client: OpenAI, mem: JevalMemory, question: str, task: str) -> str:
    context = mem.memory_retrieve(question, top_k=5)
    if not context:
        context = mem.full_memory[:3000]
    prompt = ANSWER_PROMPT.format(context=context, task=task[:300], question=question)
    return _complete(client, prompt, max_tokens=300)


def _judge(client: OpenAI, question: str, reference: str, predicted: str) -> tuple[bool, str]:
    prompt = JUDGE_PROMPT.format(
        question=question, reference=reference, predicted=predicted
    )
    raw = _complete(client, prompt, max_tokens=150)
    try:
        clean = raw.strip().strip("```json").strip("```").strip()
        result = json.loads(clean)
        return bool(result["correct"]), result.get("reasoning", "")
    except Exception:
        correct = any(w in raw.lower() for w in ["true", "correct", "yes"])
        return correct, raw[:100]


# ── Episode loader ────────────────────────────────────────────────────────────

def _load_episode(dataset_name: str, split: str, idx: int) -> Dict[str, Any]:
    from datasets import load_dataset  # type: ignore

    ds = load_dataset(dataset_name, split=split)
    if idx >= len(ds):
        raise IndexError(f"episode_idx {idx} out of range — split has {len(ds)} episodes")
    return ds[idx]


# ── Predictor loading ─────────────────────────────────────────────────────────

def _load_predictor(ckpt: Optional[str]):
    """Return a loaded PreLNTransformerPredictor or None if path absent/unspecified."""
    if not ckpt or not Path(ckpt).exists():
        return None
    import torch
    from jeval.encoders.predictor_head import PreLNTransformerPredictor
    from jeval.encoders.sentence_encoder import FrozenEncoder

    enc  = FrozenEncoder()
    pred = PreLNTransformerPredictor(enc.dim())
    pred.load_state_dict(torch.load(ckpt, map_location="cpu"))
    pred.eval()
    return pred


# ── Main episode run ──────────────────────────────────────────────────────────

def run_episode(
    episode_idx: int,
    dataset_name: str,
    split: str,
    predictor_path: Optional[str],
    out_path: Path,
) -> None:
    ep = _load_episode(dataset_name, split, episode_idx)

    ep_id   = ep["episode_id"]
    task    = ep.get("task", "")
    qas     = ep["qa_pairs"] if isinstance(ep["qa_pairs"], list) else json.loads(ep["qa_pairs"])
    traj    = ep["trajectory"] if isinstance(ep["trajectory"], list) else json.loads(ep["trajectory"])

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

    # Build memory — pass trained predictor if available
    predictor = _load_predictor(predictor_path)
    mem = JevalMemory()
    if predictor is not None:
        mem.compressor = AdaptiveCompressor(
            predictor=predictor,
            backend=ExtractiveBackend(),
        )
    mem.memory_construction(traj_text, task=task)

    compressed_tokens = len(mem.full_memory.split())
    compression_ratio = compressed_tokens / max(orig_tokens, 1)

    client = _client()
    per_question: List[dict] = []
    scores: List[float] = []

    for qa in qas:
        question   = qa["question"]
        reference  = qa["answer"]
        qa_type    = qa.get("type", "")
        q_uuid     = qa.get("question_uuid", "")

        answer          = _answer(client, mem, question, task)
        correct, reason = _judge(client, question, reference, answer)
        score           = 1.0 if correct else 0.0
        scores.append(score)

        per_question.append({
            "question_uuid": q_uuid,
            "type":          qa_type,
            "question":      question,
            "answer":        answer,
            "score":         score,
            "reasoning":     reason,
        })

    result = {
        "episode_idx":       episode_idx,
        "episode_id":        ep_id,
        "score":             float(sum(scores) / max(len(scores), 1)),
        "n_questions":       len(qas),
        "per_question":      per_question,
        "compression_ratio": compression_ratio,
        "original_tokens":   orig_tokens,
        "compressed_tokens": compressed_tokens,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(
        f"episode {episode_idx}  score={result['score']:.2f}  "
        f"n_q={len(qas)}  ratio={compression_ratio:.2f}  → {out_path}"
    )


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-idx", type=int,  required=True)
    parser.add_argument("--dataset",     default="AMA-bench/AMA-bench")
    parser.add_argument("--split",       default="test")
    parser.add_argument("--predictor",   default=None)
    parser.add_argument("--out",         required=True)
    args = parser.parse_args()

    out_path = Path(args.out)

    try:
        run_episode(
            episode_idx=args.episode_idx,
            dataset_name=args.dataset,
            split=args.split,
            predictor_path=args.predictor,
            out_path=out_path,
        )
    except Exception:
        tb = traceback.format_exc()
        error_result = {
            "episode_idx": args.episode_idx,
            "error":       str(sys.exc_info()[1]),
            "traceback":   tb,
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(error_result, indent=2))
        print(f"ERROR episode {args.episode_idx}: {sys.exc_info()[1]}", file=sys.stderr)
        # Exit 0 so SLURM marks the task succeeded; aggregate step handles errors
        sys.exit(0)


if __name__ == "__main__":
    main()
