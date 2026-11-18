"""Assign train/val/test splits by work, stratified by domain (manga pages vs landscape images).

Usage: python make_split.py --manifest /workspace/data/proc/manifest.jsonl
"""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

HOLDOUT_FRACTION = {"test": 0.07, "val": 0.05}
MIN_WORKS_FOR_WORK_SPLIT = 8


def domain_of(row: dict) -> str:
    return "landscape" if row.get("kind") == "landscape" else "manga"


def stable_order(work: int) -> str:
    return hashlib.md5(f"work-{work}".encode()).hexdigest()


def split_by_work(works: dict[int, list[dict]]) -> dict[str, str]:
    """Whole works go to test, then val, until each reaches its share of usable pages."""
    usable_total = sum(r["ok"] for rows in works.values() for r in rows)
    assignment: dict[str, str] = {}
    pending = sorted(works, key=stable_order)
    for split, fraction in HOLDOUT_FRACTION.items():
        taken = 0
        while pending and taken < fraction * usable_total:
            work = pending.pop(0)
            taken += sum(r["ok"] for r in works[work])
            assignment.update({r["name"]: split for r in works[work]})
    assignment.update({r["name"]: "train" for work in pending for r in works[work]})
    return assignment


def split_by_block(works: dict[int, list[dict]]) -> dict[str, str]:
    """Too few works to hold any out: use the trailing pages of each work instead."""
    assignment: dict[str, str] = {}
    for rows in works.values():
        test_from = int(len(rows) * (1 - HOLDOUT_FRACTION["test"]))
        val_from = int(len(rows) * (1 - HOLDOUT_FRACTION["test"] - HOLDOUT_FRACTION["val"]))
        for index, row in enumerate(rows):
            assignment[row["name"]] = "test" if index >= test_from else "val" if index >= val_from else "train"
    return assignment


def assign_splits(rows: list[dict]) -> list[dict]:
    """Return new rows with `domain` and `split` set; input rows are not modified."""
    by_domain: dict[str, dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_domain[domain_of(row)][row["work"]].append(row)
    assignment: dict[str, str] = {}
    for works in by_domain.values():
        splitter = split_by_work if len(works) >= MIN_WORKS_FOR_WORK_SPLIT else split_by_block
        assignment.update(splitter(works))
    return [{**row, "domain": domain_of(row), "split": assignment[row["name"]]} for row in rows]


def count_table(rows: list[dict]) -> dict:
    table: dict = defaultdict(lambda: defaultdict(lambda: {"pages_ok": 0, "regions_ok": 0, "works": set()}))
    for row in rows:
        if not row["ok"]:
            continue
        cell = table[row["domain"]][row["split"]]
        cell["pages_ok"] += 1
        cell["regions_ok"] += row["regions"]
        cell["works"].add(row["work"])
    return {
        domain: {split: {**cell, "works": len(cell["works"])} for split, cell in splits.items()}
        for domain, splits in table.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    rows = assign_splits(rows)
    args.manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    print(json.dumps(count_table(rows), indent=2))


if __name__ == "__main__":
    main()
