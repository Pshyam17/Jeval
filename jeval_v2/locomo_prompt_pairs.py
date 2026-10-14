"""Render blinded, paired LoCoMo prompts from fixed-memory score rows."""
import argparse
import json
import random
from pathlib import Path


PROMPT = """Answer the question using the memory below. The risk percentile estimates how much source information the memory may have lost; it is a heuristic, not evidence. If the memory does not support an answer, say you do not know. Do not infer missing facts from the percentile.

Compression risk percentile: {risk:.1f} / 100

Memory:
{memory}

Question: {question}
Answer:"""


def percentiles(values):
    """Midranks make score scales comparable without harm labels."""
    n = len(values)
    return [100 * (sum(v < value for v in values) + 0.5 * sum(v == value for v in values)) / n
            for value in values]


def build(rows, seed=0):
    ids = [str(row["id"]) for row in rows]
    if not rows or len(set(ids)) != len(rows):
        raise ValueError("rows must be nonempty and have unique question IDs")
    required = ("question", "compressed_memory", "cosine", "epe")
    for row in rows:
        if any(field not in row for field in required):
            raise ValueError(f"{row['id']}: missing required prompt field")
        if not row["compressed_memory"].strip():
            raise ValueError(f"{row['id']}: compressed_memory is empty")
    ranks = {arm: percentiles([float(row[arm]) for row in rows]) for arm in ("cosine", "epe")}
    rng = random.Random(seed)
    prompts, key, review = [], [], []
    for i, row in enumerate(rows):
        arms = ["cosine", "epe"]
        rng.shuffle(arms)
        labels = dict(zip(("A", "B"), arms))
        for side, arm in labels.items():
            prompts.append({"id": str(row["id"]), "side": side,
                            "prompt": PROMPT.format(risk=ranks[arm][i], memory=row["compressed_memory"],
                                                    question=row["question"])})
            key.append({"id": str(row["id"]), "side": side, "arm": arm,
                        "raw_score": float(row[arm]), "risk_percentile": ranks[arm][i]})
        review.append({"id": str(row["id"]), "reference_answer": row.get("answer"),
                       "answer_A": "", "answer_B": "", "A_correct": None, "B_correct": None,
                       "A_supported": None, "B_supported": None, "notes": ""})
    return prompts, key, review


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="JSONL with id, question, compressed_memory, cosine, epe")
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.input).read_text(encoding="utf-8").splitlines() if line.strip()]
    prompts, key, review = build(rows, args.seed)
    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix, content in (("prompts", prompts), ("key", key), ("review", review)):
        path = prefix.with_name(prefix.name + f".{suffix}.jsonl")
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in content), encoding="utf-8")
    print(json.dumps({"questions": len(rows), "prompts": len(prompts), "prefix": str(prefix)}))


if __name__ == "__main__":
    main()
