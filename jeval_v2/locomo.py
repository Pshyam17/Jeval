"""LoCoMo best-of-N pipeline: segments, conversation folds, candidate pairs, selection, QA scoring."""
import argparse
import json
import random
import re
import string
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

SKIP_CATEGORY = 5  # adversarial questions have no gold answer
ARMS = ("first", "random", "cosine", "epe")
COMPARISONS = (("epe", "cosine"), ("epe", "random"), ("cosine", "random"), ("epe", "first"))


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path, rows):
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def pair_id(segment_id, cand):
    return f"{segment_id}#{cand}"


def sessions(sample):
    conversation = sample["conversation"]
    numbers = sorted(int(m.group(1)) for key in conversation if (m := re.fullmatch(r"session_(\d+)", key)))
    for number in numbers:
        yield number, conversation.get(f"session_{number}_date_time", ""), conversation[f"session_{number}"]


def render_turn(turn):
    text = f"{turn['speaker']}: {turn['text']}"
    if turn.get("blip_caption"):
        text += f" [shared image: {turn['blip_caption']}]"
    return text


def segment_sample(sample, turns):
    sample_id = str(sample["sample_id"])
    segments = []
    for number, date, session in sessions(sample):
        for start in range(0, len(session), turns):
            window = session[start:start + turns]
            text = "\n".join([f"[Session {number}, {date}]"] + [render_turn(t) for t in window])
            segments.append({"id": f"{sample_id}:s{number}:t{start}", "group_id": sample_id,
                             "order": len(segments), "dia_ids": [t["dia_id"] for t in window],
                             "original": text, "words": len(text.split())})
    return segments


def fold_splits(groups, folds, fold):
    """Whole conversations per split: test fold, next fold validation, rest train."""
    groups = sorted(set(groups))
    if folds < 3 or folds > len(groups):
        raise ValueError(f"need 3 <= folds <= conversations ({len(groups)}), got {folds}")
    if not 0 <= fold < folds:
        raise ValueError(f"fold must be in [0, {folds})")
    return {group: ("test" if i % folds == fold else "validation" if i % folds == (fold + 1) % folds else "train")
            for i, group in enumerate(groups)}


def build_pairs(segments, candidates, folds, fold):
    split = fold_splits([s["group_id"] for s in segments], folds, fold)
    by_id = {s["id"]: s for s in segments}
    pairs = []
    for cand in candidates:
        seg = by_id[cand["segment_id"]]
        pairs.append({"id": pair_id(cand["segment_id"], cand["cand"]), "group_id": seg["group_id"],
                      "split": split[seg["group_id"]], "original": seg["original"], "compressed": cand["text"]})
    return pairs


def eligible(pool, budget_words, tolerance):
    """Candidates inside the length band, so neither score can win by being longer."""
    low, high = budget_words * (1 - tolerance), budget_words * (1 + tolerance)
    inside = [c for c in pool if low <= len(c["text"].split()) <= high]
    return (inside, False) if inside else (pool, True)


def select(segments, candidates, scores, ratio, tolerance, seed):
    by_segment = defaultdict(list)
    for cand in candidates:
        by_segment[cand["segment_id"]].append(cand)
    rng = random.Random(seed)
    chosen = {arm: {} for arm in ARMS}
    fallbacks = agree = 0
    for seg in segments:
        pool = sorted(by_segment.get(seg["id"], []), key=lambda c: c["cand"])
        if not pool:
            raise ValueError(f"no candidates for segment {seg['id']}")
        pool, fell = eligible(pool, ratio * seg["words"], tolerance)
        fallbacks += fell
        for cand in pool:
            if pair_id(seg["id"], cand["cand"]) not in scores:
                raise ValueError(f"missing score for {pair_id(seg['id'], cand['cand'])}; "
                                 "every fold's test scores are required")
        score = lambda cand, key: scores[pair_id(seg["id"], cand["cand"])][key]
        chosen["first"][seg["id"]] = pool[0]
        chosen["random"][seg["id"]] = rng.choice(pool)
        chosen["cosine"][seg["id"]] = min(pool, key=lambda c: score(c, "cosine"))
        chosen["epe"][seg["id"]] = min(pool, key=lambda c: score(c, "epe"))
        agree += chosen["cosine"][seg["id"]]["cand"] == chosen["epe"][seg["id"]]["cand"]
    memories = {"original": defaultdict(list)}
    for arm in ARMS:
        memories[arm] = defaultdict(list)
    for seg in segments:  # segments are already in conversation order
        memories["original"][seg["group_id"]].append(seg["original"])
        for arm in ARMS:
            memories[arm][seg["group_id"]].append(chosen[arm][seg["id"]]["text"])
    original_words = sum(s["words"] for s in segments)
    stats = {"segments": len(segments), "band_fallback_segments": fallbacks,
             "cosine_epe_same_choice_fraction": agree / len(segments),
             "word_ratio": {arm: sum(len(c["text"].split()) for c in chosen[arm].values()) / original_words
                            for arm in ARMS}}
    return {arm: {g: "\n\n".join(parts) for g, parts in groups.items()} for arm, groups in memories.items()}, stats


def normalize(text):
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(word for word in text.split() if word not in {"a", "an", "the"})


def f1(prediction, gold):
    pred, ref = normalize(prediction).split(), normalize(gold).split()
    overlap = sum((Counter(pred) & Counter(ref)).values())
    if not pred or not ref or not overlap:
        return 0.0
    precision, recall = overlap / len(pred), overlap / len(ref)
    return 2 * precision * recall / (precision + recall)


def score_answer(prediction, gold, category):
    """Token F1 following the LoCoMo conventions for categories 1 and 3 (no stemming)."""
    gold = str(gold)
    if category == 3:
        gold = gold.split(";")[0].strip()
    if category == 1:
        predictions = [p.strip() for p in prediction.split(",")]
        return float(np.mean([max(f1(p, g.strip()) for p in predictions) for g in gold.split(",")]))
    return f1(prediction, gold)


def report(rows, bootstrap=1000, seed=0):
    by_arm = defaultdict(dict)
    for row in rows:
        by_arm[row["arm"]][row["id"]] = row
    arms = sorted(by_arm)
    reference = set(by_arm[arms[0]])
    if any(set(by_arm[arm]) != reference for arm in arms):
        raise ValueError("arms answered different question sets")
    questions = sorted(reference)
    groups = sorted({by_arm[arms[0]][q]["group_id"] for q in questions})
    group_index = {g: i for i, g in enumerate(groups)}

    def vector(arm):
        return np.array([by_arm[arm][q]["f1"] for q in questions])

    member = np.array([group_index[by_arm[arms[0]][q]["group_id"]] for q in questions])
    counts = np.bincount(member, minlength=len(groups))
    draws = np.random.default_rng(seed).integers(0, len(groups), size=(bootstrap, len(groups)))
    result = {"questions": len(questions), "conversations": len(groups), "mean_f1": {}, "f1_by_category": {},
              "paired_difference": {}}
    categories = np.array([by_arm[arms[0]][q]["category"] for q in questions])
    for arm in arms:
        values = vector(arm)
        result["mean_f1"][arm] = float(values.mean())
        result["f1_by_category"][arm] = {str(c): float(values[categories == c].mean()) for c in sorted(set(categories))}
    for left, right in COMPARISONS:
        if left not in by_arm or right not in by_arm:
            continue
        diff = vector(left) - vector(right)
        sums = np.bincount(member, weights=diff, minlength=len(groups))
        samples = sums[draws].sum(1) / counts[draws].sum(1)
        result["paired_difference"][f"{left}_minus_{right}"] = {
            "mean_f1_difference": float(diff.mean()),
            "group_bootstrap_95pct": np.quantile(samples, [0.025, 0.975]).tolist()}
    return result


def run_segment(args):
    samples = json.loads(Path(args.input).read_text(encoding="utf-8"))
    segments = [seg for sample in samples for seg in segment_sample(sample, args.turns)]
    write_jsonl(args.output, segments)
    print(json.dumps({"conversations": len(samples), "segments": len(segments),
                      "max_words": max(s["words"] for s in segments)}))


def run_make_pairs(args):
    pairs = build_pairs(read_jsonl(args.segments), read_jsonl(args.candidates), args.folds, args.fold)
    write_jsonl(args.output, pairs)
    print(json.dumps({"fold": args.fold, "pairs": len(pairs), "by_split": dict(Counter(p["split"] for p in pairs))}))


def run_select(args):
    scores = {}
    for path in args.scores:
        for row in read_jsonl(path):
            if row["id"] in scores:
                raise ValueError(f"pair {row['id']} scored in more than one fold")
            scores[row["id"]] = row
    memories, stats = select(read_jsonl(args.segments), read_jsonl(args.candidates), scores,
                             args.ratio, args.tolerance, args.seed)
    out = Path(args.output_dir)
    write_jsonl(out / "memories.jsonl", [{"arm": arm, "group_id": group, "memory": text}
                                         for arm, groups in memories.items() for group, text in sorted(groups.items())])
    (out / "selection_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


def run_report(args):
    rows = []
    for row in read_jsonl(args.predictions):
        rows.append({"arm": row["arm"], "id": row["id"], "group_id": row["group_id"], "category": row["category"],
                     "f1": score_answer(row["prediction"], row["gold"], row["category"])})
    print(json.dumps(report(rows, args.bootstrap, args.seed), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("segment", help="cut official LoCoMo conversations into fixed-turn segments")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--turns", type=int, default=6)
    p.set_defaults(run=run_segment)
    p = sub.add_parser("make-pairs", help="flatten candidates into pair JSONL with conversation-level splits")
    p.add_argument("--segments", required=True)
    p.add_argument("--candidates", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--fold", type=int, required=True)
    p.set_defaults(run=run_make_pairs)
    p = sub.add_parser("select", help="pick one candidate per segment under each rule and assemble memories")
    p.add_argument("--segments", required=True)
    p.add_argument("--candidates", required=True)
    p.add_argument("--scores", nargs="+", required=True, help="evaluate output from every fold")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--ratio", type=float, required=True, help="same target ratio used for compress")
    p.add_argument("--tolerance", type=float, default=0.25, help="allowed relative deviation from the word budget")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(run=run_select)
    p = sub.add_parser("report", help="score QA predictions and compare arms")
    p.add_argument("--predictions", required=True)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(run=run_report)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
