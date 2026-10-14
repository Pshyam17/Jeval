"""Normalize official benchmark question/evidence metadata without inventing labels."""
import argparse
import json
from pathlib import Path


def longmemeval(records):
    for item in records:
        sessions = item["haystack_sessions"]
        session_ids = item["haystack_session_ids"]
        if len(sessions) != len(session_ids):
            raise ValueError(f"{item['question_id']}: session ID/content mismatch")
        evidence_turns = []
        for session_id, turns in zip(session_ids, sessions):
            for index, turn in enumerate(turns):
                if turn.get("has_answer") is True:
                    evidence_turns.append({"session_id": str(session_id), "turn_index": index})
        yield {"dataset": "longmemeval_s_cleaned", "id": str(item["question_id"]),
               "group_id": str(item["question_id"]), "question": item["question"],
               "answer": item["answer"], "category": item["question_type"],
               "question_date": item.get("question_date"),
               "evidence_session_ids": [str(s) for s in item["answer_session_ids"]],
               "evidence_turns": evidence_turns}


def locomo(records):
    for sample in records:
        sample_id = str(sample["sample_id"])
        for index, qa in enumerate(sample["qa"]):
            yield {"dataset": "locomo10", "id": f"{sample_id}:{index}",
                   "group_id": sample_id, "question": qa["question"],
                   "answer": qa["answer"], "category": qa.get("category"),
                   "evidence_dialog_ids": qa.get("evidence", [])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["longmemeval", "locomo"], required=True)
    parser.add_argument("--input", required=True, help="official local dataset JSON")
    parser.add_argument("--output", required=True, help="normalized question manifest JSONL")
    args = parser.parse_args()
    with Path(args.input).open(encoding="utf-8") as handle:
        records = json.load(handle)
    convert = longmemeval if args.dataset == "longmemeval" else locomo
    dest = Path(args.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with dest.open("w", encoding="utf-8") as handle:
        for row in convert(records):
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    print(json.dumps({"dataset": args.dataset, "questions": count, "output": str(dest)}))


if __name__ == "__main__":
    main()
