"""Score Laya (github.com/NandhaKishorM/laya) on the unridden use case suites.

Laya is a library, not a server: `laya.Router().predict(state, questions)` runs
one non-autoregressive forward pass over an encoder checkpoint and returns the
same `{"model", "answers", "usage"}` payload unridden returns, with the same
per-question keys (laya/agent.py:322-368). The suites therefore go in unchanged
and every answer is scored by run_usecase_suites' own scorer. Suite files are
loaded, validated and hashed before the checkpoint is built, exactly as the
unridden driver does, so the labels are frozen against the same bytes.

What this driver records, and does not paper over:

* Laya's context window is the checkpoint's `max_len` (1024 for
  `typed-decisions`) against unridden's 2048, and its option head is capped at
  `head_max_len` (256). `build_sequence` silently truncates the state, clips any
  option past 48 tokens, and clips the instructions to whatever the options
  leave (laya/common.py:61-86). This driver replays that accounting with Laya's
  own tokenizer for every question and writes the counts into each observation
  row and into `laya-extras.json`. Nothing is dropped and nothing is retuned.
* Laya rounds every reported probability to four decimals
  (laya/agent.py:342,353,359). They are kept exactly as reported rather than
  renormalised, so a distribution can sum to 1 plus or minus about 1e-4.
* Laya's confidence is normalised entropy, `1 - H(p)/log(k)`
  (laya/common.py:210-216), not unridden's max label probability. The answers
  handed to `score_answer` carry unridden's own max-probability confidence so
  the scorer is like for like; Laya's native value is kept beside it in the
  observation row as `laya_confidence`.
* The `typed-decisions` checkpoint ships `choice:11+` temperature 0.1006, which
  Laya itself refuses and clamps to 0.5 while warning that the affected buckets
  are uncalibrated (laya/common.py:224-240, laya/agent.py:208-220). The warning
  is captured into run.json rather than suppressed.
* Laya returns no logits. The `raw_label_logits` in each observation row are the
  natural log of its reported probabilities, present only so
  compare_observations.py can run. They are not comparable to unridden's raw
  label logits, and the row says so.

Usage:
    python scripts/unridden/bench_competitor_laya.py \
        --suites unridden/examples/usecases \
        --out outputs/competitor-laya-v1 \
        --checkpoint typed-decisions
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
import warnings
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

from unridden.api.schema import (
    Answer,
    ChoiceAnswer,
    ChoiceQuestion,
    NoulAnswer,
    NoulQuestion,
    Question,
    ScoreAnswer,
    ScoreQuestion,
)

Row = dict[str, Any]

PROBABILITY_SEMANTICS = "laya_temperature_scaled_marker_logit_softmax_rounded_4dp_v1"
LOGIT_SEMANTICS = "natural_log_of_reported_probabilities_v1"
LOGIT_NOTE = (
    "Laya returns no logits. These are ln(p) of its reported, temperature "
    "scaled probabilities, written only so compare_observations.py can run. "
    "They are not unridden raw label logits and must not be compared as such."
)
CONFIDENCE_NOTE = (
    "answers[].confidence is unridden's max label probability, recomputed from "
    "Laya's distribution so the scorer is like for like. Laya's own confidence "
    "(normalised entropy 1 - H(p)/log(k), laya/common.py:210-216) is in "
    "diagnostics[].laya_confidence."
)
# laya/common.py:68 caps each rendered option at 48 tokens before the budget
# arithmetic starts.
OPTION_TOKEN_CAP = 48


class RequestFailed(RuntimeError):
    """Laya refused or failed a case. Evidence, not a reason to stop."""


def laya_question(question: Question) -> Row:
    """One unridden question in the shape `Agent.system_one` documents.

    Laya reads `type`, `instructions` and `criteria` (laya/agent.py:255-263,
    266-277); every suite question supplies all three, so nothing is invented
    here. Choice criteria pass through as the mapping Laya accepts, score
    criteria as the ordered list, and a noul rubric as the `true`/`false`
    mapping `render_options` reads (laya/common.py:41-46).
    """
    if isinstance(question, ChoiceQuestion):
        return {
            "type": "choice",
            "instructions": question.instructions,
            "criteria": dict(question.criteria),
        }
    if isinstance(question, ScoreQuestion):
        return {
            "type": "score",
            "instructions": question.instructions,
            "criteria": list(question.criteria),
        }
    assert isinstance(question, NoulQuestion)
    rubric = question.criteria
    return {
        "type": "noul",
        "instructions": question.instructions,
        "criteria": (
            None if rubric is None else {"true": rubric.true, "false": rubric.false}
        ),
    }


def budget_report(
    tokenizer: Any, state: Any, definition: Row, max_len: int, head_max_len: int
) -> Row:
    """What Laya's own `build_sequence` would keep and drop for this question.

    Replays the arithmetic of laya/common.py:61-86 with the checkpoint's own
    tokenizer so the truncation is counted rather than assumed away. Nothing
    here feeds the model; it only measures what the model was given.
    """
    from laya.agent import Agent  # type: ignore[import-not-found]
    from laya.common import (  # type: ignore[import-not-found]
        render_options,
        serialize_state,
    )

    internal = Agent._to_internal(definition)
    mask = tokenizer.mask_token
    options = render_options(internal)
    head_text = "{} question: {}".format(
        internal["t"], str(internal["ins"]).replace(mask, " ")
    )
    head_full = tokenizer(head_text, add_special_tokens=False)["input_ids"]
    rendered = [
        tokenizer(" " + option.replace(mask, " "), add_special_tokens=False)[
            "input_ids"
        ]
        for option in options
    ]
    option_ids = [[0] + item[:OPTION_TOKEN_CAP] for item in rendered]
    over_cap = sum(len(item) > OPTION_TOKEN_CAP for item in rendered)
    budget = head_max_len - sum(len(item) for item in option_ids)
    shared_clip = 0
    if budget < 16:
        per = max(4, (head_max_len - 16) // max(1, len(option_ids)))
        clipped = [item[:per] for item in option_ids]
        shared_clip = sum(
            len(short) < len(long)
            for short, long in zip(clipped, option_ids, strict=True)
        )
        option_ids = clipped
        budget = head_max_len - sum(len(item) for item in option_ids)
    head_ids = head_full[: max(8, budget)]
    prefix = 1 + len(head_ids) + 1
    markers = []
    position = prefix
    for item in option_ids:
        markers.append(position)
        position += len(item)
    length = position + 1
    room = max(0, max_len - length - 1)
    state_ids = tokenizer(
        serialize_state(state).replace(mask, " "), add_special_tokens=False
    )["input_ids"]
    return {
        "options": len(options),
        "state_tokens": len(state_ids),
        "state_tokens_kept": min(len(state_ids), room),
        "state_tokens_dropped": max(0, len(state_ids) - room),
        "state_truncated": len(state_ids) > room,
        "state_room": room,
        "instruction_tokens": len(head_full),
        "instruction_tokens_kept": len(head_ids),
        "instruction_clipped": len(head_ids) < len(head_full),
        "options_over_48_token_cap": over_cap,
        "options_clipped_by_head_budget": shared_clip,
        "option_clipped": bool(over_cap or shared_clip),
        "markers_lost_to_max_len": sum(marker >= max_len for marker in markers),
    }


def map_answers(case: Case, body: Row) -> tuple[dict[str, Answer], Row]:
    """Laya's answers as unridden Answer objects, plus per-question diagnostics."""
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
            # unridden orders a noul's labels true, false (compiler.py:83), so
            # the diagnostic vector follows that order on both sides.
            probabilities = [p_true, 1.0 - p_true]
            keys = ["true", "false"]
            reported = {"true": p_true, "false": 1.0 - p_true}
            answers[qid] = NoulAnswer(noul=p_true)
        else:
            distribution = raw.get("probabilities")
            if not isinstance(distribution, dict):
                raise RequestFailed(f"{qid!r}: answer carries no probabilities")
            if question.type == "choice":
                keys = list(question.criteria)
            else:
                assert isinstance(question, ScoreQuestion)
                keys = [str(index) for index in range(len(question.criteria))]
            missing = [key for key in keys if key not in distribution]
            if missing:
                raise RequestFailed(f"{qid!r}: probabilities missing {missing}")
            probabilities = [float(distribution[key]) for key in keys]
            reported = dict(zip(keys, probabilities, strict=True))
            total = math.fsum(probabilities)
            if not math.isfinite(total) or not 0.99 <= total <= 1.01:
                raise RequestFailed(f"{qid!r}: probabilities sum to {total}")
            mapped = dict(zip(keys, probabilities, strict=True))
            confidence = max(probabilities)
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
            "raw_label_logits": [
                math.log(max(value, 1e-12)) for value in probabilities
            ],
            "logit_semantics": LOGIT_SEMANTICS,
            "logit_note": LOGIT_NOTE,
            "probability_semantics": PROBABILITY_SEMANTICS,
            "reported_probabilities": reported,
            "laya_confidence": raw.get("confidence"),
            "laya_confidence_formula": "normalised_entropy_1_minus_H_over_log_k_v1",
            "laya_act_probability": (raw.get("action") or {}).get("act_probability"),
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


def expected_calibration_error(
    confidences: Sequence[float], correct: Sequence[bool], bins: int = 15
) -> float | None:
    """Binned |confidence - accuracy|, the statistic laya/common.py:197-207 uses."""
    if not confidences:
        return None
    edges = [index / bins for index in range(bins + 1)]
    error = 0.0
    for low, high in zip(edges[:-1], edges[1:], strict=True):
        members = [
            (value, flag)
            for value, flag in zip(confidences, correct, strict=True)
            if (value > low or (low == 0.0 and value == 0.0)) and value <= high
        ]
        if not members:
            continue
        share = len(members) / len(confidences)
        error += share * abs(
            statistics.fmean([value for value, _ in members])
            - statistics.fmean([float(flag) for _, flag in members])
        )
    return error


def coverage_table(confidences: Sequence[float], correct: Sequence[bool]) -> list[Row]:
    """Accuracy over the most confident fraction of answers, at fixed coverages."""
    if not confidences:
        return []
    order = sorted(range(len(confidences)), key=lambda index: -confidences[index])
    table: list[Row] = []
    for coverage in (0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0):
        take = max(1, round(coverage * len(order)))
        selected = order[:take]
        table.append(
            {
                "coverage": coverage,
                "n": take,
                "accuracy": statistics.fmean(
                    [float(correct[index]) for index in selected]
                ),
                "confidence_threshold": confidences[selected[-1]],
            }
        )
    return table


def calibration_block(confidences: Sequence[float], correct: Sequence[bool]) -> Row:
    return {
        "n": len(confidences),
        "accuracy": (
            statistics.fmean([float(flag) for flag in correct]) if correct else None
        ),
        "mean_confidence": (statistics.fmean(confidences) if confidences else None),
        "ece_15_bin": expected_calibration_error(confidences, correct),
        "auroc_for_correctness": auroc(confidences, correct),
        "accuracy_at_coverage": coverage_table(confidences, correct),
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


def reference_calibration(path: Path) -> Row | None:
    """The same calibration block for a unridden observations.jsonl, if given."""
    if path is None or not path.is_file():
        return None
    confidences: list[float] = []
    correct: list[bool] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            for outcome in row.get("outcomes", []):
                confidences.append(float(outcome["confidence"]))
                correct.append(bool(outcome["correct"]))
    block = calibration_block(confidences, correct)
    block["source"] = str(path)
    block["note"] = (
        "Computed from the unridden reference run's own observation rows, with "
        "the same confidence definition (max label probability) on both sides."
    )
    return block


def budget_totals(reports: Sequence[Row]) -> Row:
    if not reports:
        return {"questions": 0}
    dropped = [int(item["state_tokens_dropped"]) for item in reports]
    return {
        "questions": len(reports),
        "state_truncated_questions": sum(
            bool(item["state_truncated"]) for item in reports
        ),
        "state_tokens_dropped_total": sum(dropped),
        "state_tokens_dropped_max": max(dropped),
        "instruction_clipped_questions": sum(
            bool(item["instruction_clipped"]) for item in reports
        ),
        "option_clipped_questions": sum(
            bool(item["option_clipped"]) for item in reports
        ),
        "options_over_48_token_cap_total": sum(
            int(item["options_over_48_token_cap"]) for item in reports
        ),
        "options_clipped_by_head_budget_total": sum(
            int(item["options_clipped_by_head_budget"]) for item in reports
        ),
        "questions_over_20_options": sum(int(item["options"]) > 20 for item in reports),
        "max_options": max(int(item["options"]) for item in reports),
        "markers_lost_to_max_len": sum(
            int(item["markers_lost_to_max_len"]) for item in reports
        ),
    }


def build_router(args: argparse.Namespace) -> tuple[Any, Any, Row]:
    """The Router, the loaded Agent, and what was loaded, on Laya's own defaults."""
    from laya import Router  # type: ignore[import-not-found]

    notes: list[str] = []
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        router = Router(device=args.device)
        agent = router.load(args.checkpoint)
        notes.extend(str(item.message) for item in captured)
    decision = router.route("probe", {}, model=args.checkpoint)
    profile: Row = {
        "checkpoint": args.checkpoint,
        "hub_repo": decision["repo"],
        "router_reason": decision["reason"],
        "model_name": agent.cfg.get("model_name"),
        "encoder": agent.cfg.get("encoder"),
        "max_len": agent.cfg.get("max_len", 512),
        "head_max_len": agent.cfg.get("head_max_len", 192),
        "device": str(agent.device),
        "dtype": str(agent.dtype),
        "temperature_raw": agent.temperature_raw,
        "temperature_applied": agent.temperature,
        "temperature_by_options_raw": agent.temperature_by_options_raw,
        "temperature_by_options_applied": agent.temperature_by_options,
        "load_warnings": notes,
    }
    if args.max_len is not None:
        # The matched-budget control Laya runs on itself
        # (research/scripts/build_benchmark_nb.py:574-584): cap max_len and
        # re-run, isolating the context window from the rest.
        profile["max_len_override"] = args.max_len
        agent.cfg["max_len"] = args.max_len
        profile["max_len"] = args.max_len
    return router, agent, profile


def run(args: argparse.Namespace) -> int:
    args.out.mkdir(parents=True, exist_ok=False)
    observations = args.out / "observations.jsonl"
    manifest: Row = {
        "state": "starting",
        "competitor": "laya",
        "repository": "https://github.com/NandhaKishorM/laya",
        "transport": (
            "in-process library call, one laya.Router.predict per case "
            "(all of a case's questions in one forward pass)"
        ),
        "started_unix": time.time(),
    }
    # Labels are frozen before the checkpoint is built: validate, then hash.
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
    if args.provenance is not None:
        manifest["weights_provenance"] = json.loads(
            args.provenance.read_text(encoding="utf-8")
        )

    started = time.perf_counter()
    router, agent, profile = build_router(args)
    manifest["load_seconds"] = time.perf_counter() - started
    manifest["laya"] = profile
    manifest["state"] = "running"
    write_json(args.out / "run.json", manifest)

    outcomes: list[Outcome] = []
    rankings: list[RankOutcome] = []
    errors: list[Row] = []
    latencies: list[float] = []
    native_confidences: list[float | None] = []
    budgets: list[Row] = []
    scored_budgets: list[Row] = []
    input_tokens = 0
    index = 0
    max_len = int(profile["max_len"])
    head_max_len = int(profile["head_max_len"])
    for suite in suites:
        for case in suite.cases:
            questions = {
                qid: laya_question(question)
                for qid, question in case.request.questions.items()
            }
            reports = {
                qid: budget_report(
                    agent.tok,
                    case.request.state,
                    definition,
                    max_len,
                    head_max_len,
                )
                for qid, definition in questions.items()
            }
            budgets.extend(reports.values())
            body: Row | None = None
            failure: str | None = None
            request_started = time.perf_counter()
            try:
                body = router.predict(
                    case.request.state, questions, model=args.checkpoint
                )
                answers, diagnostics = map_answers(case, body)
            except (RequestFailed, ValueError, RuntimeError, KeyError) as error:
                # A refused case is evidence, not a reason to stop.
                failure = f"{type(error).__name__}: {error}"
            elapsed_ms = (time.perf_counter() - request_started) * 1000
            record: Row = {
                "index": index,
                "suite": suite.slug,
                "case": case.name,
                "difficulty": case.difficulty,
                "rationale": case.rationale,
                "request": case.request.model_dump(mode="json"),
                "laya_request": {"state": case.request.state, "questions": questions},
                "budget": reports,
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
                        "questions": len(case.targets),
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
                    diagnostics[target.qid]["laya_confidence"]
                    for target in case.targets
                )
                scored_budgets.extend(reports[target.qid] for target in case.targets)
                record["status"] = "ok"
                record["response"] = {
                    "model": body.get("model"),
                    "answers": {
                        qid: answer.model_dump(mode="json")
                        for qid, answer in answers.items()
                    },
                    "usage": {
                        "input_tokens": int(usage.get("input_tokens") or 0),
                        "output_tokens": int(usage.get("output_tokens") or 0),
                    },
                    "diagnostics": diagnostics,
                    "routing": body.get("routing"),
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

    posed = int(manifest["questions"])
    correct = sum(item.correct for item in outcomes)
    unridden_confidence = [item.confidence for item in outcomes]
    was_correct = [item.correct for item in outcomes]
    paired = [
        (value, flag)
        for value, flag in zip(native_confidences, was_correct, strict=True)
        if value is not None
    ]
    write_json(
        args.out / "laya-extras.json",
        {
            "answered": {
                "questions_posed": posed,
                "questions_scored": len(outcomes),
                "questions_unanswered": posed - len(outcomes),
                "accuracy_scored_only": (correct / len(outcomes) if outcomes else None),
                "accuracy_unanswered_counted_wrong": correct / posed if posed else None,
                "note": (
                    "summary.json scores only the questions Laya answered, the "
                    "convention run_usecase_suites uses. "
                    "accuracy_unanswered_counted_wrong is the strict figure over "
                    "every question the suites pose."
                ),
            },
            "calibration_max_probability": calibration_block(
                unridden_confidence, was_correct
            ),
            "calibration_laya_native_confidence": calibration_block(
                [value for value, _ in paired], [flag for _, flag in paired]
            ),
            "calibration_unridden_reference": reference_calibration(args.reference),
            "brier": brier_block(outcomes),
            "budget_all_questions": budget_totals(budgets),
            "budget_scored_questions": budget_totals(scored_budgets),
            "probability_semantics": PROBABILITY_SEMANTICS,
            "confidence_note": CONFIDENCE_NOTE,
            "latency_note": (
                "summary.json request_latency_ms is wall clock around one "
                "in-process laya.Router.predict call per case, which answers "
                "every question of the case in one forward pass. The unridden "
                "reference run measures in-process calls to an owned llama.cpp "
                "child, which answers a case one question at a time. The two "
                "run on different hardware: see run.json laya.device."
            ),
        },
    )
    manifest["state"] = "complete"
    manifest["errored_cases"] = len(errors)
    manifest["finished_unix"] = time.time()
    write_json(args.out / "run.json", manifest)
    router.unload()
    accuracy = summary["overall"]["accuracy"]
    print(
        "LAYA_RUN_COMPLETE",
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
    parser.add_argument(
        "--checkpoint",
        default="typed-decisions",
        help="Laya checkpoint name passed to Router.predict(model=...)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="torch device for the Agent; Laya's default is automatic",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=None,
        help="override the checkpoint's context window, Laya's own control",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=None,
        help="a unridden observations.jsonl to compute the same calibration block for",
    )
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
