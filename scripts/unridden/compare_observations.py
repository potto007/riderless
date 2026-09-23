"""Compare two use-case suite runs question by question.

Reports answer changes, the largest probability and raw logit moves, and the
token and latency totals. Used to check that a configuration change either
reproduces a baseline bit for bit or to quantify how far it moved.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections.abc import Iterator
from pathlib import Path
from typing import Any

Row = dict[str, Any]


def load(path: Path) -> Iterator[Row]:
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def answer_of(answer: Row) -> Any:
    if answer["type"] == "choice":
        return answer["choice"]
    if answer["type"] == "noul":
        return answer["noul"] >= 0.5
    return max(answer["probabilities"], key=answer["probabilities"].__getitem__)


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def compare(left_path: Path, right_path: Path) -> Row:
    left = {(row["suite"], row["case"]): row for row in load(left_path)}
    right = {(row["suite"], row["case"]): row for row in load(right_path)}
    if set(left) != set(right):
        raise ValueError("the two runs do not cover the same cases")
    changed: list[Row] = []
    probability_moves: list[float] = []
    logit_moves: list[float] = []
    questions = 0
    identical_logits = True
    for key, old in left.items():
        new = right[key]
        if old["status"] != "ok" or new["status"] != "ok":
            raise ValueError(f"{key} did not succeed in both runs")
        for qid, old_answer in old["response"]["answers"].items():
            questions += 1
            new_answer = new["response"]["answers"][qid]
            old_probabilities = (
                {"true": old_answer["noul"]}
                if old_answer["type"] == "noul"
                else old_answer["probabilities"]
            )
            new_probabilities = (
                {"true": new_answer["noul"]}
                if new_answer["type"] == "noul"
                else new_answer["probabilities"]
            )
            move = max(
                abs(value - new_probabilities[label])
                for label, value in old_probabilities.items()
            )
            probability_moves.append(move)
            old_raw = old["response"]["diagnostics"][qid]["raw_label_logits"]
            new_raw = new["response"]["diagnostics"][qid]["raw_label_logits"]
            logit_move = max(abs(a - b) for a, b in zip(old_raw, new_raw, strict=True))
            logit_moves.append(logit_move)
            if old_raw != new_raw:
                identical_logits = False
            if answer_of(old_answer) != answer_of(new_answer):
                changed.append(
                    {
                        "suite": key[0],
                        "case": key[1],
                        "question": qid,
                        "before": answer_of(old_answer),
                        "after": answer_of(new_answer),
                        "probability_move": move,
                    }
                )
    latencies = {
        name: [row["latency_ms"] for row in rows.values()]
        for name, rows in (("before", left), ("after", right))
    }
    return {
        "left": str(left_path),
        "right": str(right_path),
        "questions": questions,
        "bit_identical_logits": identical_logits,
        "changed_answers": len(changed),
        "maximum_probability_move": max(probability_moves, default=0.0),
        "questions_moved_over_0_3": sum(move > 0.3 for move in probability_moves),
        "maximum_raw_logit_move": max(logit_moves, default=0.0),
        "changes": changed,
        "latency_ms": {
            name: {
                "median": statistics.median(values),
                "p95": percentile(values, 0.95),
                "total": sum(values),
            }
            for name, values in latencies.items()
        },
        "input_tokens": {
            name: sum(row["response"]["usage"]["input_tokens"] for row in rows.values())
            for name, rows in (("before", left), ("after", right))
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = compare(args.left, args.right)
    if args.out is not None:
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    summary = {key: value for key, value in report.items() if key != "changes"}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
