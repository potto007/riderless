"""Score Kev (github.com/jaredpalmer/kev) on the riderless use case suites.

Kev serves the same TypeSafe System One request shape riderless serves, so the
suites go over the wire unchanged (POST <base-url>/v1/systemone) and every
answer is scored by run_usecase_suites' own scorer. The suite files are loaded,
validated and hashed before the first request, exactly as the riderless driver
does, so the labels are frozen against the same bytes.

Three facts about Kev's answers shape what this driver records, and none of them
is papered over:

* Kev rounds every probability to two decimals (kev/api.py:132,140-146). The
  rounded vector is renormalised here, which is what Kev's own RemotePredictor
  does to any System One endpoint (kev/predictors.py:57-60). Brier, the
  reliability table and any confidence threshold are therefore quantised
  relative to a riderless run.
* Kev's Choice confidence is (p_max - 1/K) / (1 - 1/K) and its Score confidence
  is a modal-distance statistic (kev/api.py:120-129), neither of which is
  riderless's max label probability. To keep the scorer like for like, the
  answers handed to score_answer carry riderless's own max-probability
  confidence, recomputed from Kev's distribution; Kev's native value is kept
  beside it in the observation row as `kev_confidence`.
* Kev returns no logits. The `raw_label_logits` in each observation row are the
  natural log of Kev's reported (renormalised) probabilities, present only so
  compare_observations.py can run. They are not comparable to riderless's raw
  label logits, and the row says so.

Usage:
    python scripts/riderless/bench_competitor_kev.py \
        --suites riderless/examples/usecases \
        --out outputs/competitor-kev-v1 \
        --base-url http://127.0.0.1:8009
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from run_usecase_suites import (  # type: ignore[import-not-found]
    Case,
    Outcome,
    RankOutcome,
    build_summary,
    load_suites,
    score_answer,
    score_ranking,
    write_json,
)

from riderless.api.schema import (
    Answer,
    ChoiceAnswer,
    NoulAnswer,
    ScoreAnswer,
    ScoreQuestion,
)

Row = dict[str, Any]

PROBABILITY_SEMANTICS = "kev_two_decimal_probabilities_renormalised_v1"
LOGIT_SEMANTICS = "natural_log_of_reported_probabilities_v1"
LOGIT_NOTE = (
    "Kev returns no logits. These are ln(p) of its two-decimal, renormalised "
    "probabilities, written only so compare_observations.py can run. They are "
    "not riderless raw label logits and must not be compared as such."
)
CONFIDENCE_NOTE = (
    "answers[].confidence is riderless's max label probability, recomputed from "
    "Kev's distribution so the scorer is like for like. Kev's own confidence "
    "((p_max - 1/K)/(1 - 1/K) for choice, modal distance for score) is in "
    "diagnostics[].kev_confidence."
)


class RequestFailed(RuntimeError):
    """The endpoint refused or failed a case. Evidence, not a reason to stop."""


def renormalise(values: Sequence[float]) -> list[float]:
    total = math.fsum(values)
    if not math.isfinite(total) or total <= 0:
        raise RequestFailed(f"probability vector does not normalise: {list(values)}")
    return [float(value) / total for value in values]


def log_probabilities(probabilities: Sequence[float]) -> list[float]:
    floor = 1e-12
    return [math.log(max(float(value), floor)) for value in probabilities]


def kev_payload(case: Case, model: str) -> Row:
    payload: Row = case.request.model_dump(mode="json")
    payload["model"] = model
    return payload


def post(
    base_url: str, payload: Row, timeout: float, retries: int
) -> tuple[Row, float]:
    body = json.dumps(payload, allow_nan=False).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/systemone",
        data=body,
        method="POST",
        headers={"content-type": "application/json"},
    )
    last: Exception | None = None
    for attempt in range(retries):
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                decoded = json.loads(response.read())
            return decoded, (time.perf_counter() - started) * 1000
        except urllib.error.HTTPError as error:
            # A 4xx is the server's real verdict on this request; do not retry it.
            detail = error.read().decode("utf-8", "replace")[:500]
            if error.code < 500:
                raise RequestFailed(f"HTTP {error.code}: {detail}") from error
            last = error
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as error:
            last = error
        time.sleep(2**attempt)
    raise RequestFailed(f"endpoint failed after {retries} attempts: {last}")


def map_answers(case: Case, body: Row) -> tuple[dict[str, Answer], Row]:
    """Kev's answers as riderless Answer objects, plus per-question diagnostics."""
    answers: dict[str, Answer] = {}
    diagnostics: Row = {}
    served = body.get("answers")
    if not isinstance(served, dict):
        raise RequestFailed("response has no answers object")
    for qid, question in case.request.questions.items():
        raw = served.get(qid)
        if not isinstance(raw, dict):
            raise RequestFailed(f"response is missing an answer for {qid!r}")
        if raw.get("type") != question.type:
            raise RequestFailed(
                f"{qid!r}: answer type {raw.get('type')!r} is not {question.type!r}"
            )
        if question.type == "noul":
            p_true = float(raw["noul"])
            if not 0.0 <= p_true <= 1.0:
                raise RequestFailed(f"{qid!r}: noul {p_true} outside [0, 1]")
            probabilities = [p_true, 1.0 - p_true]
            keys = ["true", "false"]
            answers[qid] = NoulAnswer(noul=p_true)
            native = None
        else:
            reported = raw.get("probabilities")
            if not isinstance(reported, dict):
                raise RequestFailed(f"{qid!r}: answer carries no probabilities")
            if question.type == "choice":
                keys = list(question.criteria)
            else:
                assert isinstance(question, ScoreQuestion)
                keys = [str(index) for index in range(len(question.criteria))]
            missing = [key for key in keys if key not in reported]
            if missing:
                raise RequestFailed(f"{qid!r}: probabilities missing {missing}")
            probabilities = renormalise([float(reported[key]) for key in keys])
            mapped = dict(zip(keys, probabilities, strict=True))
            confidence = max(probabilities)
            native = raw.get("confidence")
            if question.type == "choice":
                choice = raw.get("choice")
                if choice not in mapped:
                    raise RequestFailed(f"{qid!r}: choice {choice!r} is not an option")
                answers[qid] = ChoiceAnswer(
                    choice=str(choice), probabilities=mapped, confidence=confidence
                )
            else:
                assert isinstance(question, ScoreQuestion)
                answers[qid] = ScoreAnswer(
                    score=float(raw["score"]),
                    legend=dict(zip(keys, question.criteria, strict=True)),
                    probabilities=mapped,
                    confidence=confidence,
                )
        diagnostics[qid] = {
            "raw_label_logits": log_probabilities(probabilities),
            "logit_semantics": LOGIT_SEMANTICS,
            "logit_note": LOGIT_NOTE,
            "probability_semantics": PROBABILITY_SEMANTICS,
            "reported_probabilities": (
                {"true": float(raw["noul"]), "false": 1.0 - float(raw["noul"])}
                if question.type == "noul"
                else {key: float(raw["probabilities"][key]) for key in keys}
            ),
            "kev_confidence": native,
            "kev_confidence_formula": {
                "choice": "rescaled_max_probability_(p_max-1/K)/(1-1/K)_v1",
                "score": "modal_distance_v1",
                "noul": None,
            }[question.type],
            "confidence_note": CONFIDENCE_NOTE,
            "generated_tokens": 0,
        }
    return answers, diagnostics


def auroc(scores: Sequence[float], labels: Sequence[bool]) -> float | None:
    """Rank-based AUROC of `scores` for predicting label True, ties averaged."""
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


def confidence_block(
    outcomes: Sequence[Outcome], native: Sequence[float | None]
) -> Row:
    correct = [outcome.correct for outcome in outcomes]
    riderless_style = [outcome.confidence for outcome in outcomes]
    paired = [
        (value, flag)
        for value, flag in zip(native, correct, strict=True)
        if value is not None
    ]
    return {
        "n": len(outcomes),
        "auroc_max_probability": auroc(riderless_style, correct),
        "auroc_kev_native_confidence": (
            auroc([value for value, _ in paired], [flag for _, flag in paired])
            if paired
            else None
        ),
        "n_with_native_confidence": len(paired),
        "note": (
            "auroc_max_probability uses riderless's confidence formula recomputed "
            "from Kev's distribution and is the like-for-like number. "
            "auroc_kev_native_confidence uses Kev's own rescaled statistic and "
            "covers choice and score questions only."
        ),
    }


def brier_block(outcomes: Sequence[Outcome]) -> Row:
    noul = [item for item in outcomes if item.kind == "noul"]
    choice = [item for item in outcomes if item.kind == "choice"]
    score = [item for item in outcomes if item.kind == "score"]

    def multiclass(items: Sequence[Outcome], gold: Any) -> float | None:
        if not items:
            return None
        totals = []
        for item in items:
            target = str(gold(item))
            totals.append(
                math.fsum(
                    (value - (1.0 if key == target else 0.0)) ** 2
                    for key, value in item.probabilities.items()
                )
            )
        return statistics.fmean(totals)

    return {
        "noul_brier": (
            statistics.fmean(
                [
                    (float(item.p_true or 0.0) - float(bool(item.expected))) ** 2
                    for item in noul
                ]
            )
            if noul
            else None
        ),
        "choice_multiclass_brier": multiclass(choice, lambda item: item.expected),
        "score_multiclass_brier": multiclass(score, lambda item: int(item.expected)),
        "note": (
            "noul_brier matches summary.json overall.noul.brier. The multiclass "
            "figures are sum over labels of (p - y)^2, one per question."
        ),
    }


def run(args: argparse.Namespace) -> int:
    args.out.mkdir(parents=True, exist_ok=False)
    observations = args.out / "observations.jsonl"
    manifest: Row = {
        "state": "starting",
        "competitor": "kev",
        "repository": "https://github.com/jaredpalmer/kev",
        "base_url": args.base_url,
        "requested_model": args.model,
        "transport": "HTTP POST /v1/systemone (kev.serve), one request per case",
        "started_unix": time.time(),
    }
    # Labels are frozen before the first request: validate, then hash.
    suites = load_suites(args.suites)
    manifest["suites"] = [
        {
            "suite": suite.slug,
            "path": str(suite.path),
            "sha256": suite.sha256,
            "cases": len(suite.cases),
        }
        for suite in suites
    ]
    manifest["cases"] = sum(len(suite.cases) for suite in suites)
    manifest["questions"] = sum(
        len(case.targets) for suite in suites for case in suite.cases
    )
    try:
        with urllib.request.urlopen(
            f"{args.base_url.rstrip('/')}/v1/models", timeout=args.timeout
        ) as response:
            manifest["served_models"] = json.loads(response.read())
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as error:
        print(f"KEV_RUN_FAILED could not reach {args.base_url}: {error}", flush=True)
        manifest["state"] = "failed"
        manifest["error"] = str(error)
        write_json(args.out / "run.json", manifest)
        return 2
    if args.provenance is not None:
        manifest["weights_provenance"] = json.loads(
            args.provenance.read_text(encoding="utf-8")
        )
    manifest["state"] = "running"
    write_json(args.out / "run.json", manifest)

    outcomes: list[Outcome] = []
    rankings: list[RankOutcome] = []
    errors: list[Row] = []
    latencies: list[float] = []
    native_confidences: list[float | None] = []
    input_tokens = 0
    index = 0
    for suite in suites:
        for case in suite.cases:
            payload = kev_payload(case, args.model)
            body: Row | None = None
            failure: str | None = None
            started = time.perf_counter()
            try:
                body, _ = post(args.base_url, payload, args.timeout, args.retries)
                answers, diagnostics = map_answers(case, body)
            except RequestFailed as error:
                failure = f"{type(error).__name__}: {error}"
            elapsed_ms = (time.perf_counter() - started) * 1000
            record: Row = {
                "index": index,
                "suite": suite.slug,
                "case": case.name,
                "difficulty": case.difficulty,
                "rationale": case.rationale,
                "request": payload,
                "latency_ms": elapsed_ms,
            }
            index += 1
            if failure is not None:
                record["status"] = "error"
                record["error"] = failure
                if body is not None:
                    record["raw_response"] = body
                errors.append(
                    {
                        "suite": suite.slug,
                        "case": case.name,
                        "difficulty": case.difficulty,
                        "error": failure,
                    }
                )
            else:
                assert body is not None
                latencies.append(elapsed_ms)
                usage = body.get("usage") or {}
                input_tokens += int(usage.get("input_tokens") or 0)
                case_outcomes = [
                    score_answer(suite.slug, case, target, answers[target.qid])
                    for target in case.targets
                ]
                case_rankings = [
                    score_ranking(suite.slug, case, ranking, answers)
                    for ranking in case.rankings
                ]
                outcomes.extend(case_outcomes)
                rankings.extend(case_rankings)
                native_confidences.extend(
                    diagnostics[target.qid]["kev_confidence"] for target in case.targets
                )
                record["status"] = "ok"
                record["response"] = {
                    "model": body.get("model"),
                    "answers": {
                        qid: answer.model_dump(mode="json")
                        for qid, answer in answers.items()
                    },
                    "usage": {
                        "input_tokens": int(usage.get("input_tokens") or 0),
                        "output_tokens": 0,
                    },
                    "diagnostics": diagnostics,
                    "kev_reported_latency_ms": body.get("latency_ms"),
                    "kev_reported_output_tokens": (usage.get("output_tokens")),
                }
                record["outcomes"] = [item.as_row() for item in case_outcomes]
                record["rankings"] = [item.as_row() for item in case_rankings]
            with observations.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
        print("SUITE_COMPLETE", suite.slug, flush=True)

    summary = build_summary(
        suites,
        outcomes,
        rankings,
        errors,
        latencies,
        {"generated_tokens": 0, "input_tokens": input_tokens},
    )
    write_json(args.out / "summary.json", summary)
    write_json(
        args.out / "kev-extras.json",
        {
            "confidence": confidence_block(outcomes, native_confidences),
            "brier": brier_block(outcomes),
            "probability_semantics": PROBABILITY_SEMANTICS,
            "latency_note": (
                "summary.json request_latency_ms is wall clock around the HTTP "
                "POST from this process to kev.serve on localhost, so it "
                "includes JSON, HTTP and FastAPI overhead. The riderless "
                "reference run measures in-process calls to an owned child "
                "process. observations rows also carry Kev's own "
                "kev_reported_latency_ms (model time only)."
            ),
        },
    )
    manifest["state"] = "complete"
    manifest["errored_cases"] = len(errors)
    manifest["finished_unix"] = time.time()
    write_json(args.out / "run.json", manifest)
    accuracy = summary["overall"]["accuracy"]
    print(
        "KEV_RUN_COMPLETE",
        f"questions={len(outcomes)}",
        f"accuracy={accuracy if accuracy is None else round(accuracy, 4)}",
        f"errored_cases={len(errors)}",
        flush=True,
    )
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suites", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8009")
    parser.add_argument("--model", default="kev-latest")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument(
        "--provenance",
        type=Path,
        default=None,
        help="JSON file of weight hashes and revisions to copy into run.json",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
