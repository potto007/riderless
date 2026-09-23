"""Measure what state snapshots cost and save on the split18-30-v1 worker.

Gate 8 of the state-snapshot design: creation, durable save, disk load, first
promotion, host restore and warm (resident) branching, each as p50/p95 over
repeats, plus serialized bytes, at several prefix lengths. Each branch is
compared with a stock full-prompt pass on the same tokens (the `reference`
command), which is what answering without a snapshot costs.

Timings are the worker's own measured milliseconds; wall time of the JSONL
round trip is reported separately. Needs the GPU and the user's confirmation.

    python scripts/riderless/bench_snapshots.py --bundle build/snapshot-worker \\
        --model <gguf> --gpu --out outputs/snapshots
"""

from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any

from scripts.riderless.qualify_snapshots import (
    ANSWER_PREFIX,
    PROMPT_VERSION,
    Worker,
    context_messages,
    decision_messages,
)

Row = dict[str, Any]
FILLER = (
    "The support log records one customer message per line, each with a time, "
    "a channel, and a short summary of what was asked or promised. "
)
QUESTION = (
    "QUESTION:\nChoose the single best option for the state.\n\nOPTIONS:\n"
    "A: billing - A payment or refund problem\nB: technical - A software defect\n\n"
    "Reply with one option label only."
)


def percentile(values: list[float], share: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(share * (len(ordered) - 1)))]


def stats(values: list[float]) -> Row:
    return {
        "n": len(values),
        "p50": percentile(values, 0.5),
        "p95": percentile(values, 0.95),
        "mean": statistics.fmean(values),
    }


def run(args: argparse.Namespace) -> Row:
    client = Worker(args.bundle, args.model, gpu=args.gpu, reference=True)
    results: list[Row] = []
    scratch = Path(tempfile.mkdtemp(prefix="riderless-bench-"))
    try:
        for target in args.prefix_tokens:
            state = FILLER * max(1, target // 30)
            messages, content_bytes = context_messages(state)
            question = {
                "id": "q",
                "messages": decision_messages(state, QUESTION),
                "answer_prefix": ANSWER_PREFIX,
                "labels": client.labels[:2],
                "prompt_version": PROMPT_VERSION,
                "save_as": None,
            }
            timings: dict[str, list[float]] = {
                key: []
                for key in (
                    "create_18_30",
                    "create_18",
                    "promote",
                    "save",
                    "load",
                    "branch_resident",
                    "branch_host",
                    "branch_disk",
                    "stock_full_prompt",
                )
            }
            sizes: Row = {}
            tokens = 0
            for repeat in range(args.repeats):
                tag = f"p{target}_{repeat}"
                mark = time.perf_counter()
                created = client.call(
                    {
                        "type": "create",
                        "messages": messages,
                        "answer_prefix": "",
                        "freeze": {"kind": "context", "content_bytes": content_bytes},
                        "checkpoints": {"18": f"{tag}_18", "30": f"{tag}_30"},
                        "labels": None,
                        "top_logits": 0,
                    }
                )
                timings["create_18_30"].append(created["timing_ms"]["total"])
                tokens = int(created["tokens"])
                sizes = {
                    row["snapshot_id"][-2:]: row["bytes"]
                    for row in created["snapshots"]
                }
                branch = client.call(
                    {
                        "type": "evaluate",
                        "snapshot_id": f"{tag}_30",
                        "readout_blocks": 30,
                        "questions": [question],
                    }
                )["questions"][0]
                assert branch["snapshot"]["restore"] == "resident"
                timings["branch_resident"].append(branch["timing_ms"])
                solo = client.call(
                    {
                        "type": "create",
                        "messages": messages,
                        "answer_prefix": "",
                        "freeze": {"kind": "context", "content_bytes": content_bytes},
                        "checkpoints": {"18": f"{tag}_solo"},
                        "labels": None,
                        "top_logits": 0,
                    }
                )
                timings["create_18"].append(solo["timing_ms"]["total"])
                promoted = client.call(
                    {
                        "type": "promote",
                        "snapshot_id": f"{tag}_solo",
                        "new_id": f"{tag}_prom",
                    }
                )
                timings["promote"].append(promoted["timing_ms"]["total"])
                branch = client.call(
                    {
                        "type": "evaluate",
                        "snapshot_id": f"{tag}_30",
                        "readout_blocks": 30,
                        "questions": [question],
                    }
                )["questions"][0]
                assert branch["snapshot"]["restore"] == "host"
                timings["branch_host"].append(branch["timing_ms"])
                directory = scratch / tag
                directory.mkdir()
                mark = time.perf_counter()
                saved = client.call(
                    {
                        "type": "save",
                        "snapshot_id": f"{tag}_30",
                        "directory": str(directory),
                    }
                )
                timings["save"].append((time.perf_counter() - mark) * 1000)
                files = {
                    name: entry["sha256"] for name, entry in saved["files"].items()
                }
                mark = time.perf_counter()
                client.call(
                    {
                        "type": "load",
                        "snapshot_id": f"{tag}_disk",
                        "directory": str(directory),
                        "files": files,
                    }
                )
                timings["load"].append((time.perf_counter() - mark) * 1000)
                branch = client.call(
                    {
                        "type": "evaluate",
                        "snapshot_id": f"{tag}_disk",
                        "readout_blocks": 30,
                        "questions": [question],
                    }
                )["questions"][0]
                timings["branch_disk"].append(branch["timing_ms"])
                stock = client.call(
                    {
                        "type": "reference",
                        "messages": question["messages"],
                        "answer_prefix": ANSWER_PREFIX,
                        "labels": question["labels"],
                        "top_logits": 0,
                    }
                )
                timings["stock_full_prompt"].append(stock["timing_ms"])
                for suffix in ("_18", "_30", "_solo", "_prom", "_disk"):
                    client.call({"type": "drop", "snapshot_id": tag + suffix})
            row: Row = {
                "prefix_tokens": tokens,
                "suffix_tokens": branch["snapshot"]["suffix_tokens"],
                "bytes": sizes,
                "ms": {key: stats(values) for key, values in timings.items()},
            }
            print(
                json.dumps(
                    {
                        "prefix_tokens": tokens,
                        **{k: round(v["p50"], 2) for k, v in row["ms"].items()},
                    }
                ),
                flush=True,
            )
            results.append(row)
    finally:
        client.close()
    return {"profile": "split18-30-v1", "repeats": args.repeats, "results": results}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument(
        "--prefix-tokens", type=int, nargs="+", default=[128, 512, 1024, 1800]
    )
    parser.add_argument("--out", type=Path, default=Path("outputs/snapshots"))
    args = parser.parse_args()
    args.bundle = args.bundle.resolve()
    args.model = args.model.resolve()
    result = run(args)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"bench-{time.strftime('%Y%m%dT%H%M%S')}.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(path)


if __name__ == "__main__":
    main()
