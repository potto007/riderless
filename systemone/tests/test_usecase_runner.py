"""CPU tests for the use case suite runner and its scorer.

Every test uses a deterministic fake backend. No test loads a model or touches
the GPU.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pytest

from scripts.systemone.run_usecase_suites import (
    Case,
    Ranking,
    SuiteError,
    Target,
    load_suites,
    metric_block,
    run,
    score_answer,
    score_ranking,
)
from systemone.api.app import ApiConfig
from systemone.api.backend import BackendRequestError
from systemone.api.compiler import CompiledBatch
from systemone.api.schema import (
    BackendProfile,
    ChoiceAnswer,
    NoulAnswer,
    ScoreAnswer,
    SystemOneRequest,
    WorkerBatchResult,
)

PROFILE = BackendProfile(
    model_id="local-gemma-systemone-v1",
    model_name="fixture",
    model_sha256="a" * 64,
    runtime_sha256="b" * 64,
    labels=["A", "B", "C", "D"],
    label_token_ids=[11, 12, 13, 14],
    context_size=2048,
    batch_size=256,
    ubatch_size=256,
    threads=8,
    max_questions=32,
    generated_tokens=0,
    callbacks_enabled=False,
    execution_mode="full",
)


class FakeBackend:
    """Replays fixed probabilities and can reject one named question."""

    def __init__(
        self,
        probabilities: dict[str, list[float]],
        *,
        reject: str | None = None,
    ) -> None:
        self.probabilities = probabilities
        self.reject = reject
        self.profile: BackendProfile | None = None
        self.ready = False
        self.started = 0
        self.closed = 0

    async def start(self) -> BackendProfile:
        self.started += 1
        self.profile = PROFILE
        self.ready = True
        return PROFILE

    async def close(self) -> None:
        self.closed += 1
        self.ready = False

    async def evaluate(
        self, batch: CompiledBatch, *, timeout: float
    ) -> WorkerBatchResult:
        del timeout
        rows = []
        for index, question in enumerate(batch.questions):
            if question.id == self.reject:
                raise BackendRequestError("request exceeds model limits")
            count = len(question.labels)
            values = self.probabilities.get(question.id, [1.0 / count] * count)
            assert len(values) == count, question.id
            rows.append(
                {
                    "id": question.id,
                    "label_logits": [math.log(value) for value in values],
                    "label_token_ids": PROFILE.label_token_ids[:count],
                    "allowed_label_mass": 0.75,
                    "full_vocabulary_argmax": {"token_id": 12, "logit": 4.0},
                    "prompt_sha256": f"{index + 1:064x}",
                    "prompt_tokens": 10 + index,
                    "processed_tokens": 10 + index,
                    "reused_tokens": 0,
                    "cache_cleared": True,
                    "timing_ms": 1.0,
                }
            )
        return WorkerBatchResult.model_validate(
            {
                "type": "result",
                "id": "fixture",
                "model_sha256": PROFILE.model_sha256,
                "runtime_sha256": PROFILE.runtime_sha256,
                "generated_tokens": 0,
                "callbacks_enabled": False,
                "execution_mode": "full",
                "questions": rows,
            }
        )


def make_case(name: str = "case") -> Case:
    return Case(
        name=name,
        difficulty="easy",
        rationale="fixture",
        request=SystemOneRequest.model_validate(
            {"state": "x", "questions": {"q": {"type": "noul", "instructions": "x"}}}
        ),
        targets=(),
        rankings=(),
    )


def suite_document() -> dict[str, Any]:
    return {
        "suite": "fixture",
        "family": "Fixture family",
        "source": ["fixture cookbook"],
        "description": "A runner fixture, not a capability claim.",
        "cases": [
            {
                "name": "alpha",
                "difficulty": "easy",
                "rationale": "the state names the route",
                "request": {
                    "state": "The card was charged twice.",
                    "questions": {
                        "route": {
                            "type": "choice",
                            "instructions": "Choose the route",
                            "criteria": {"billing": None, "technical": "a defect"},
                        },
                        "duplicate": {
                            "type": "noul",
                            "instructions": "The charge repeated",
                        },
                    },
                },
                "targets": {
                    "route": {"type": "choice", "expected": "billing"},
                    "duplicate": {"type": "noul", "expected": True},
                },
            },
            {
                "name": "beta",
                "difficulty": "medium",
                "rationale": "rejected by the fixture backend",
                "request": {
                    "state": "An oversized state.",
                    "questions": {
                        "over_budget": {
                            "type": "noul",
                            "instructions": "This is rejected",
                        }
                    },
                },
                "targets": {"over_budget": {"type": "noul", "expected": True}},
            },
            {
                "name": "gamma",
                "difficulty": "hard",
                "rationale": "the state states the severity",
                "request": {
                    "state": "The outage stopped all payments.",
                    "questions": {
                        "severity": {
                            "type": "score",
                            "instructions": "Rate severity",
                            "criteria": ["low", "medium", "high"],
                        },
                        "rank_a": {"type": "noul", "instructions": "Doc A is best"},
                        "rank_b": {"type": "noul", "instructions": "Doc B is best"},
                    },
                },
                "targets": {
                    "severity": {"type": "score", "expected_level": 2, "tolerance": 1},
                    "rank_a": {"type": "noul", "expected": True},
                    "rank_b": {"type": "noul", "expected": False},
                },
                "rankings": [
                    {
                        "name": "documents",
                        "question_ids": ["rank_a", "rank_b"],
                        "expected_order": ["rank_a", "rank_b"],
                    }
                ],
            },
        ],
    }


def write_suite(directory: Path, document: dict[str, Any]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{document['suite']}.json"
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return path


def runner_args(tmp_path: Path, suites: Path) -> argparse.Namespace:
    stub = tmp_path / "stub"
    stub.write_text("stub", encoding="utf-8")
    return argparse.Namespace(
        suites=[suites],
        out=tmp_path / "out",
        worker=stub,
        manifest=stub,
        model_path=stub,
        extra_log=tmp_path / "extra.log",
        allow_gpu=True,
        batch_size=256,
        no_share_prefix=False,
    )


def test_scoring_math_uses_literal_values() -> None:
    case = make_case()
    outcomes = [
        score_answer(
            "s",
            case,
            Target(qid="c1", kind="choice", expected="a"),
            ChoiceAnswer(
                choice="a", probabilities={"a": 0.7, "b": 0.3}, confidence=0.7
            ),
        ),
        score_answer(
            "s",
            case,
            Target(qid="c2", kind="choice", expected="b"),
            ChoiceAnswer(
                choice="a", probabilities={"a": 0.6, "b": 0.4}, confidence=0.6
            ),
        ),
        score_answer("s", case, Target("n1", "noul", True), NoulAnswer(noul=0.9)),
        score_answer("s", case, Target("n2", "noul", False), NoulAnswer(noul=0.4)),
        score_answer("s", case, Target("n3", "noul", False), NoulAnswer(noul=0.8)),
        score_answer(
            "s",
            case,
            Target(qid="s1", kind="score", expected=2),
            ScoreAnswer(
                score=1.6,
                legend={"0": "low", "1": "mid", "2": "high"},
                probabilities={"0": 0.1, "1": 0.2, "2": 0.7},
                confidence=0.7,
            ),
        ),
        score_answer(
            "s",
            case,
            Target(qid="s2", kind="score", expected=1, tolerance=1),
            ScoreAnswer(
                score=0.7,
                legend={"0": "low", "1": "mid", "2": "high"},
                probabilities={"0": 0.5, "1": 0.3, "2": 0.2},
                confidence=0.5,
            ),
        ),
    ]

    assert [item.got for item in outcomes] == ["a", "a", True, False, True, 2, 0]
    assert [item.correct for item in outcomes] == [
        True,
        False,
        True,
        True,
        False,
        True,
        False,
    ]
    assert outcomes[3].confidence == pytest.approx(0.6)
    assert outcomes[6].within_tolerance is True

    block = metric_block(outcomes, [])
    assert block["questions"] == 7
    assert block["correct"] == 4
    assert block["accuracy"] == pytest.approx(4 / 7)
    assert block["choice"] == {"n": 2, "correct": 1, "accuracy": pytest.approx(0.5)}
    assert block["noul"]["accuracy"] == pytest.approx(2 / 3)
    assert block["noul"]["brier"] == pytest.approx(0.27)
    assert block["noul"]["mean_p_true_on_true_targets"] == pytest.approx(0.9)
    assert block["noul"]["mean_p_true_on_false_targets"] == pytest.approx(0.6)
    assert block["score"]["exact_accuracy"] == pytest.approx(0.5)
    assert block["score"]["within_tolerance_accuracy"] == pytest.approx(1.0)
    assert block["score"]["mean_absolute_error"] == pytest.approx(0.35)
    assert block["confidence"]["mean_on_correct"] == pytest.approx(0.725)
    assert block["confidence"]["mean_on_incorrect"] == pytest.approx(1.9 / 3)
    assert [(row["count"], row["accuracy"]) for row in block["reliability"]] == [
        (0, None),
        (0, None),
        (1, pytest.approx(0.0)),
        (4, pytest.approx(0.75)),
        (2, pytest.approx(0.5)),
    ]
    baseline = block["majority_baseline"]
    assert baseline["choice"] == {"label": "a", "n": 2, "accuracy": pytest.approx(0.5)}
    assert baseline["noul"] == {
        "label": False,
        "n": 3,
        "accuracy": pytest.approx(2 / 3),
    }
    assert baseline["score"] == {"label": 1, "n": 2, "accuracy": pytest.approx(0.5)}
    assert baseline["combined_accuracy"] == pytest.approx(4 / 7)


def test_ranking_scores_prefix_order_and_ties() -> None:
    case = make_case()
    ranking = Ranking(
        name="documents",
        question_ids=("q1", "q2", "q3", "q4"),
        expected_order=("q1", "q2"),
    )
    answers = {
        "q1": NoulAnswer(noul=0.9),
        "q2": NoulAnswer(noul=0.7),
        "q3": NoulAnswer(noul=0.8),
        "q4": NoulAnswer(noul=0.1),
    }

    outcome = score_ranking("s", case, ranking, answers)

    assert outcome.predicted_order == ("q1", "q3", "q2", "q4")
    assert outcome.top1_hit is True
    # (q1,q2) plus each of q1 and q2 against the unranked q3 and q4.
    assert outcome.total_pairs == 5
    assert outcome.concordant_pairs == 4
    block = metric_block([], [outcome])
    assert block["ranking"]["top1_accuracy"] == pytest.approx(1.0)
    assert block["ranking"]["pairwise_agreement"] == pytest.approx(0.8)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda doc: doc["cases"][0]["targets"].pop("route"),
            "do not cover questions",
        ),
        (
            lambda doc: doc["cases"][0]["targets"].__setitem__(
                "route", {"type": "noul", "expected": True}
            ),
            "is 'noul' but the question is 'choice'",
        ),
        (
            lambda doc: doc["cases"][0]["targets"]["route"].__setitem__(
                "expected", "shipping"
            ),
            "unknown option 'shipping'",
        ),
        (
            lambda doc: doc["cases"][2]["targets"]["severity"].__setitem__(
                "expected_level", 3
            ),
            "outside 0..2",
        ),
        (
            lambda doc: doc["cases"][2]["rankings"][0]["question_ids"].__setitem__(
                1, "severity"
            ),
            "is not a noul question",
        ),
        (
            lambda doc: doc["cases"][2]["rankings"][0]["expected_order"].__setitem__(
                0, "missing"
            ),
            "is not ranked",
        ),
        (
            lambda doc: doc["cases"][1].__setitem__("name", "alpha"),
            "case names must be unique",
        ),
        (
            lambda doc: doc["cases"][0]["request"].__setitem__(
                "model", "some-other-model"
            ),
            "must not set 'model'",
        ),
        (
            lambda doc: doc["cases"][0].__setitem__("difficulty", "trivial"),
            "difficulty must be one of",
        ),
        (
            lambda doc: doc["cases"][0]["request"]["questions"]["route"].__setitem__(
                "criteria", {"only": None}
            ),
            "request is invalid",
        ),
    ],
)
def test_suite_validation_rejects_bad_labels(
    tmp_path: Path, mutate: Any, message: str
) -> None:
    document = suite_document()
    mutate(document)
    path = write_suite(tmp_path / "suites", document)

    with pytest.raises(SuiteError) as error:
        load_suites([path])

    assert message in str(error.value)


def test_load_suites_reads_cases_targets_and_hash(tmp_path: Path) -> None:
    path = write_suite(tmp_path / "suites", suite_document())

    suites = load_suites([tmp_path / "suites"])

    assert len(suites) == 1
    suite = suites[0]
    assert suite.slug == "fixture"
    assert len(suite.sha256) == 64
    assert [case.name for case in suite.cases] == ["alpha", "beta", "gamma"]
    assert [target.qid for target in suite.cases[0].targets] == ["route", "duplicate"]
    assert suite.cases[2].targets[0].tolerance == 1
    assert suite.cases[2].rankings[0].expected_order == ("rank_a", "rank_b")
    assert path.exists()


@pytest.mark.asyncio
async def test_run_records_request_error_and_keeps_going(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_suite(tmp_path / "suites", suite_document())
    args = runner_args(tmp_path, tmp_path / "suites")
    backend = FakeBackend(
        {
            "route": [0.8, 0.2],
            "duplicate": [0.95, 0.05],
            "severity": [0.1, 0.2, 0.7],
            "rank_a": [0.9, 0.1],
            "rank_b": [0.2, 0.8],
        },
        reject="over_budget",
    )

    code = await run(args, backend_factory=lambda _: backend)

    assert code == 0
    assert backend.started == 1 and backend.closed == 1
    assert "SUITE_COMPLETE fixture" in capsys.readouterr().out
    records = [
        json.loads(line)
        for line in (args.out / "observations.jsonl").read_text().splitlines()
    ]
    assert [record["case"] for record in records] == ["alpha", "beta", "gamma"]
    assert [record["status"] for record in records] == ["ok", "error", "ok"]
    assert "BackendRequestError" in records[1]["error"]
    assert records[0]["response"]["usage"]["output_tokens"] == 0

    summary = json.loads((args.out / "summary.json").read_text())
    assert summary["totals"]["cases"] == 3
    assert summary["totals"]["errored_cases"] == 1
    assert summary["totals"]["scored_questions"] == 5
    assert summary["totals"]["generated_tokens"] == 0
    assert summary["totals"]["input_tokens"] == 54
    assert summary["overall"]["accuracy"] == pytest.approx(1.0)
    assert summary["by_difficulty"]["medium"]["questions"] == 0
    assert summary["by_suite"]["fixture"]["ranking"]["top1_hits"] == 1
    assert summary["failed_questions"] == []
    assert summary["errored_cases"][0]["case"] == "beta"
    assert summary["request_latency_ms"]["n"] == 2

    manifest = json.loads((args.out / "run.json").read_text())
    assert manifest["state"] == "complete"
    assert manifest["errored_cases"] == 1
    assert manifest["suites"][0]["sha256"] == load_suites([args.suites[0]])[0].sha256
    assert manifest["questions"] == 6


@pytest.mark.asyncio
async def test_run_reports_wrong_answers_against_frozen_labels(
    tmp_path: Path,
) -> None:
    write_suite(tmp_path / "suites", suite_document())
    args = runner_args(tmp_path, tmp_path / "suites")
    backend = FakeBackend(
        {
            "route": [0.3, 0.7],
            "duplicate": [0.2, 0.8],
            "severity": [0.7, 0.2, 0.1],
            "rank_a": [0.1, 0.9],
            "rank_b": [0.6, 0.4],
        },
        reject="over_budget",
    )

    code = await run(args, backend_factory=lambda _: backend)

    assert code == 0
    summary = json.loads((args.out / "summary.json").read_text())
    failed = {row["qid"]: row for row in summary["failed_questions"]}
    assert set(failed) == {"route", "duplicate", "severity", "rank_a", "rank_b"}
    assert failed["route"]["got"] == "technical"
    assert failed["route"]["expected"] == "billing"
    assert failed["route"]["probabilities"] == {
        "billing": pytest.approx(0.3),
        "technical": pytest.approx(0.7),
    }
    assert failed["severity"]["within_tolerance"] is False
    assert failed["severity"]["difficulty"] == "hard"
    assert summary["by_suite"]["fixture"]["ranking"]["top1_hits"] == 0
    assert summary["by_suite"]["fixture"]["ranking"]["pairwise_agreement"] == 0.0
    assert summary["failed_rankings"][0]["ranking"] == "documents"


@pytest.mark.asyncio
async def test_run_validates_every_suite_before_loading_the_model(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_suite(tmp_path / "suites", suite_document())
    broken = suite_document()
    broken["suite"] = "broken"
    broken["cases"][0]["targets"]["route"]["expected"] = "shipping"
    write_suite(tmp_path / "suites", broken)
    args = runner_args(tmp_path, tmp_path / "suites")

    def factory(config: ApiConfig) -> FakeBackend:
        del config
        raise AssertionError("the model must not load when a suite is invalid")

    code = await run(args, backend_factory=factory)

    assert code == 2
    assert "USECASE_RUN_FAILED SuiteError" in capsys.readouterr().out
    manifest = json.loads((args.out / "run.json").read_text())
    assert manifest["state"] == "failed"
    assert "unknown option 'shipping'" in manifest["error"]
    assert "suites" not in manifest
    assert not (args.out / "observations.jsonl").exists()


@pytest.mark.asyncio
async def test_run_refuses_an_existing_output_directory(tmp_path: Path) -> None:
    write_suite(tmp_path / "suites", suite_document())
    args = runner_args(tmp_path, tmp_path / "suites")
    args.out.mkdir()

    with pytest.raises(FileExistsError):
        await run(args, backend_factory=lambda _: FakeBackend({}))
