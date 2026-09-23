"""Compare the sequential-fallback request across two validation runs.

The batched worker answers a request that needs more KV cells than
`batched_context` with the ADR 0002 sequential algorithm inside the batched
context. That is a third numeric regime, not a copy of the sequential worker's
output, and this script quantifies the gap: it reads the `batched_fallback`
record from two `summary.json` files and reports the largest raw label logit
move and the largest probability move, so the number quoted in the results doc
is reproducible from the evidence directories.

    python scripts/unridden/compare_fallback.py \
        --sequential outputs/gpu-validation-batched-off/summary.json \
        --batched outputs/gpu-validation-batched-on/summary.json \
        --out outputs/gpu-validation-batched-on/fallback-compare.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

Row = dict[str, Any]


def fallback_record(path: Path) -> Row:
    summary = json.loads(path.read_text(encoding="utf-8"))
    record = summary.get("batched_fallback")
    if record is None:
        raise ValueError(f"{path} has no batched_fallback record")
    return dict(record)


def answer_of(probabilities: dict[str, float]) -> str:
    return max(probabilities, key=probabilities.__getitem__)


def compare(sequential_path: Path, batched_path: Path) -> Row:
    left = fallback_record(sequential_path)
    right = fallback_record(batched_path)
    if set(left["raw_label_logits"]) != set(right["raw_label_logits"]):
        raise ValueError("the two runs did not send the same fallback questions")
    if left["prompt_tokens"] != right["prompt_tokens"]:
        raise ValueError("the fallback request differs between the two runs")
    questions: list[Row] = []
    for key in sorted(left["raw_label_logits"]):
        old_raw = left["raw_label_logits"][key]
        new_raw = right["raw_label_logits"][key]
        old_probabilities = left["probabilities"][key]
        new_probabilities = right["probabilities"][key]
        logit_move = max(abs(a - b) for a, b in zip(old_raw, new_raw, strict=True))
        probability_move = max(
            abs(value - new_probabilities[label])
            for label, value in old_probabilities.items()
        )
        questions.append(
            {
                "question": key,
                "prompt_tokens": left["prompt_tokens"][key],
                "bit_identical_logits": old_raw == new_raw,
                "raw_logit_move": logit_move,
                "probability_move": probability_move,
                "sequential_answer": answer_of(old_probabilities),
                "fallback_answer": answer_of(new_probabilities),
            }
        )
    return {
        "sequential": str(sequential_path),
        "batched": str(batched_path),
        "questions": len(questions),
        "kv_cells": right["kv_cells"],
        "batched_context": right["batched_context"],
        "evaluation": right["evaluation"],
        "answers_changed": sum(
            row["sequential_answer"] != row["fallback_answer"] for row in questions
        ),
        "bit_identical_logits": all(row["bit_identical_logits"] for row in questions),
        "maximum_raw_logit_move": max(row["raw_logit_move"] for row in questions),
        "maximum_probability_move": max(row["probability_move"] for row in questions),
        "by_question": questions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequential", type=Path, required=True)
    parser.add_argument("--batched", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = compare(args.sequential, args.batched)
    if args.out is not None:
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    summary = {key: value for key, value in report.items() if key != "by_question"}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
