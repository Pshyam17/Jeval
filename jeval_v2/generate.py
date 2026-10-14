"""vLLM stages: sample N compressions per segment, then answer LoCoMo questions from each memory."""
import argparse
import hashlib
import json
from pathlib import Path

from jeval_v2.locomo import SKIP_CATEGORY, read_jsonl, write_jsonl

COMPRESS_PROMPT = ("Compress the following conversation excerpt to about {budget} words. Keep every concrete "
                   "fact: names, dates, numbers, places, events, and who said or did what. Drop filler and "
                   "pleasantries. Keep the session header line. Output only the compressed text.\n\n{original}")
QA_PROMPT = ("Conversation memory:\n{memory}\n\nBased on the memory above, write an answer in the form of a "
             "short phrase for the following question. Answer with exact words from the memory whenever "
             "possible.\n\nQuestion: {question}\nShort answer:")


def load_llm(args):
    from vllm import LLM
    return LLM(model=args.model, revision=args.revision, max_model_len=args.max_model_len,
               enable_prefix_caching=True, seed=args.seed)


def chat(llm, prompts, sampling):
    return llm.chat([[{"role": "user", "content": prompt}] for prompt in prompts], sampling)


def compress(args):
    from vllm import SamplingParams
    segments = read_jsonl(args.segments)
    budgets = [max(5, round(args.ratio * seg["words"])) for seg in segments]
    prompts = [COMPRESS_PROMPT.format(budget=b, original=seg["original"]) for b, seg in zip(budgets, segments)]
    sampling = [SamplingParams(n=args.n, temperature=args.temperature, top_p=args.top_p,
                               max_tokens=2 * b + 16) for b in budgets]
    outputs = chat(load_llm(args), prompts, sampling)
    rows = []
    for seg, output in zip(segments, outputs):
        for cand, item in enumerate(output.outputs):
            text = item.text.strip()
            if text:
                rows.append({"segment_id": seg["id"], "cand": cand, "text": text})
    write_jsonl(args.output, rows)
    meta = {"model": args.model, "revision": args.revision, "ratio": args.ratio, "n": args.n,
            "temperature": args.temperature, "top_p": args.top_p, "seed": args.seed,
            "prompt_sha256": hashlib.sha256(COMPRESS_PROMPT.encode()).hexdigest(),
            "segments_sha256": hashlib.sha256(Path(args.segments).read_bytes()).hexdigest(),
            "candidates": len(rows)}
    Path(args.output).with_suffix(".metadata.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta))


def qa(args):
    from vllm import SamplingParams
    samples = json.loads(Path(args.locomo).read_text(encoding="utf-8"))
    questions = [{"id": f"{s['sample_id']}:{i}", "group_id": str(s["sample_id"]), "question": q["question"],
                  "gold": q["answer"], "category": q["category"]}
                 for s in samples for i, q in enumerate(s["qa"]) if q.get("category") != SKIP_CATEGORY]
    memories = read_jsonl(args.memories)
    # memory first and sorted by memory so the shared prefix is cached across questions
    jobs = [(m["arm"], m["group_id"], m["memory"], q) for m in sorted(memories, key=lambda m: (m["group_id"], m["arm"]))
            for q in questions if q["group_id"] == m["group_id"]]
    prompts = [QA_PROMPT.format(memory=memory, question=q["question"]) for _, _, memory, q in jobs]
    outputs = chat(load_llm(args), prompts, SamplingParams(temperature=0.0, max_tokens=args.max_answer_tokens))
    write_jsonl(args.output, [{"arm": arm, **q, "prediction": out.outputs[0].text.strip()}
                              for (arm, _, _, q), out in zip(jobs, outputs)])
    print(json.dumps({"predictions": len(jobs), "questions": len(questions)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("--model", required=True)
        p.add_argument("--revision", required=True, help="pinned model commit")
        p.add_argument("--max-model-len", type=int, default=32768)
        p.add_argument("--seed", type=int, default=0)

    p = sub.add_parser("compress")
    common(p)
    p.add_argument("--segments", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--ratio", type=float, required=True, help="target compressed/original word ratio")
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.95)
    p.set_defaults(run=compress)
    p = sub.add_parser("qa")
    common(p)
    p.add_argument("--locomo", required=True, help="official LoCoMo JSON")
    p.add_argument("--memories", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max-answer-tokens", type=int, default=64)
    p.set_defaults(run=qa)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
