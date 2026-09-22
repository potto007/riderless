"""Per-question agreement between two use-case suite runs, scored by outcome.

compare_observations.py answers "did the answer move". This answers "who was
right": how many questions only the left run got right, how many only the right
run, and the confidence AUROC and Brier of each side computed the same way from
each run's own probabilities.

    python scripts/riderless/compare_competitor_outcomes.py \
        --left outputs/usecase-suites-v041/observations.jsonl \
        --right outputs/competitor-kev-v1/observations.jsonl \
        --left-name riderless-v041 --right-name kev-9b \
        --out outputs/competitor-kev-v1/outcome-diff-vs-riderless-v041.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections.abc import Sequence
from pathlib import Path
from typing import Any

Row = dict[str, Any]


def load(path: Path) -> dict[tuple[str, str, str], Row]:
    outcomes: dict[tuple[str, str, str], Row] = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["status"] != "ok":
                continue
            for outcome in row["outcomes"]:
                outcomes[(row["suite"], row["case"], outcome["qid"])] = outcome
    return outcomes


def auroc(scores: Sequence[float], labels: Sequence[bool]) -> float | None:
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return None
    order = sorted(range(len(scores)), key=lambda index: scores[index])
    ranks = [0.0] * len(scores)
    index = 0
    while index < len(order):
        stop = index
        while stop + 1 < len(order) and scores[order[stop + 1]] == scores[order[index]]:
            stop += 1
        shared = (index + stop) / 2 + 1
        for position in range(index, stop + 1):
            ranks[order[position]] = shared
        index = stop + 1
    positive_rank_sum = math.fsum(
        rank for rank, label in zip(ranks, labels, strict=True) if label
    )
    return (positive_rank_sum - positives * (positives + 1) / 2) / (
        positives * negatives
    )


def side_metrics(outcomes: Sequence[Row]) -> Row:
    correct = [bool(item["correct"]) for item in outcomes]
    confidence = [float(item["confidence"]) for item in outcomes]
    noul = [item for item in outcomes if item["type"] == "noul"]
    multiclass = []
    for item in outcomes:
        gold = item["expected"]
        target = (
            "true"
            if (item["type"] == "noul" and gold is True)
            else "false"
            if item["type"] == "noul"
            else str(gold)
        )
        multiclass.append(
            math.fsum(
                (value - (1.0 if key == target else 0.0)) ** 2
                for key, value in item["probabilities"].items()
            )
        )
    return {
        "n": len(outcomes),
        "accuracy": statistics.fmean([float(flag) for flag in correct]),
        "confidence_auroc_for_correctness": auroc(confidence, correct),
        "noul_brier": (
            statistics.fmean(
                [
                    (
                        float(item["probabilities"]["true"])
                        - float(bool(item["expected"]))
                    )
                    ** 2
                    for item in noul
                ]
            )
            if noul
            else None
        ),
        "multiclass_brier_all_types": statistics.fmean(multiclass),
        "mean_confidence_on_correct": statistics.fmean(
            [c for c, flag in zip(confidence, correct, strict=True) if flag]
        ),
        "mean_confidence_on_incorrect": statistics.fmean(
            [c for c, flag in zip(confidence, correct, strict=True) if not flag]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--left-name", default="left")
    parser.add_argument("--right-name", default="right")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    left = load(args.left)
    right = load(args.right)
    shared = sorted(set(left) & set(right))
    only_left = []
    only_right = []
    both_wrong = 0
    both_right = 0
    for key in shared:
        lhs, rhs = left[key], right[key]
        if lhs["correct"] and rhs["correct"]:
            both_right += 1
        elif lhs["correct"]:
            only_left.append(
                {
                    "suite": key[0],
                    "case": key[1],
                    "question": key[2],
                    "type": lhs["type"],
                    "expected": lhs["expected"],
                    f"{args.right_name}_got": rhs["got"],
                    f"{args.right_name}_confidence": rhs["confidence"],
                }
            )
        elif rhs["correct"]:
            only_right.append(
                {
                    "suite": key[0],
                    "case": key[1],
                    "question": key[2],
                    "type": rhs["type"],
                    "expected": rhs["expected"],
                    f"{args.left_name}_got": lhs["got"],
                    f"{args.left_name}_confidence": lhs["confidence"],
                }
            )
        else:
            both_wrong += 1

    def by_suite(items: Sequence[Row]) -> Row:
        counts: Row = {}
        for item in items:
            counts[item["suite"]] = counts.get(item["suite"], 0) + 1
        return dict(sorted(counts.items()))

    report = {
        "left": {"path": str(args.left), "name": args.left_name},
        "right": {"path": str(args.right), "name": args.right_name},
        "scored_questions": len(shared),
        "both_correct": both_right,
        "both_incorrect": both_wrong,
        f"only_{args.left_name}_correct": len(only_left),
        f"only_{args.right_name}_correct": len(only_right),
        f"only_{args.left_name}_correct_by_suite": by_suite(only_left),
        f"only_{args.right_name}_correct_by_suite": by_suite(only_right),
        "metrics": {
            args.left_name: side_metrics([left[key] for key in shared]),
            args.right_name: side_metrics([right[key] for key in shared]),
        },
        f"only_{args.left_name}_correct_questions": only_left,
        f"only_{args.right_name}_correct_questions": only_right,
    }
    if args.out is not None:
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                key: value
                for key, value in report.items()
                if not key.endswith("_correct_questions")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
