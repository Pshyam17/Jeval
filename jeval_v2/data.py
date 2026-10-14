"""Strict data contract: one candidate compression per original segment."""
import json
from pathlib import Path

REQUIRED = {"id", "group_id", "original", "compressed", "split"}
SPLITS = {"train", "validation", "test"}


def read_pairs(path: str):
    rows = []
    ids = set()
    groups = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            missing = REQUIRED - item.keys()
            if missing:
                raise ValueError(f"line {line_number}: missing {sorted(missing)}")
            if item["split"] not in SPLITS:
                raise ValueError(f"line {line_number}: invalid split")
            if not all(isinstance(item[k], str) and item[k].strip() for k in REQUIRED):
                raise ValueError(f"line {line_number}: required fields must be nonempty strings")
            if item["id"] in ids:
                raise ValueError(f"line {line_number}: repeated pair id {item['id']}")
            ids.add(item["id"])
            group = item["group_id"]
            if group in groups and groups[group] != item["split"]:
                raise ValueError(f"group {group} crosses train/validation/test boundary")
            groups[group] = item["split"]
            if "harm" in item and item["harm"] not in (0, 1):
                raise ValueError(f"line {line_number}: harm must be 0 or 1")
            rows.append(item)
    if not rows:
        raise ValueError("empty pair file")
    return rows
