"""Contract tests for the isolated non-generative API."""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from riderless.api.compiler import PROMPT_VERSION, compile_request
from riderless.api.mapping import map_response, stable_softmax
from riderless.api.native.build import TESTED_LLAMA_REVISION
from riderless.api.schema import (
    BackendProfile,
    DecisionRequest,
    WorkerBatchResult,
    WorkerQuestionResult,
)

PROFILE = BackendProfile(
    model_id="local-gemma-riderless-v1",
    model_name="fixture",
    model_sha256="a" * 64,
    runtime_sha256="b" * 64,
    llama_revision=TESTED_LLAMA_REVISION,
    tested_revision=True,
    labels=["A", "B", "C", "D"],
    label_token_ids=[101, 102, 103, 104],
    context_size=2048,
    batch_size=256,
    ubatch_size=256,
    threads=8,
    max_questions=32,
    generated_tokens=0,
    callbacks_enabled=False,
    execution_mode="full",
)


def _request() -> DecisionRequest:
    return DecisionRequest.model_validate(
        {
            "state": {"ticket": "email bounced", "attempts": [1, 2]},
            "questions": {
                "route": {
                    "type": "choice",
                    "instructions": ["Pick", {"best": True}],
                    "criteria": {
                        "support": None,
                        "sales": {"when": "new business"},
                    },
                },
                "severity": {
                    "type": "score",
                    "instructions": None,
                    "criteria": ["low", {"name": "medium"}, ["high"]],
                },
                "urgent": {
                    "type": "noul",
                    "instructions": "The ticket is urgent",
                },
            },
        }
    )


def test_schema_preserves_supported_structured_values_and_order() -> None:
    request = _request()

    assert list(request.questions) == ["route", "severity", "urgent"]
    route = request.questions["route"]
    assert route.type == "choice"
    assert list(route.criteria) == ["support", "sales"]
    assert route.criteria["support"] is None
    score = request.questions["severity"]
    assert score.type == "score"
    assert score.criteria[1] == {"name": "medium"}


@pytest.mark.parametrize(
    "payload",
    [
        {"state": 7, "questions": {"q": {"type": "noul", "instructions": "x"}}},
        {"state": "x", "questions": {}},
        {
            "state": "x",
            "questions": {
                "q": {
                    "type": "choice",
                    "instructions": None,
                    "criteria": {"only": "one"},
                }
            },
        },
        {
            "state": "x",
            "questions": {
                "q": {"type": "noul", "instructions": None, "criteria": None}
            },
        },
        {
            "state": {"bad": math.inf},
            "questions": {"q": {"type": "noul", "instructions": "x"}},
        },
        {
            "state": "x",
            "questions": {"q": {"type": "noul", "instructions": "x", "extra": 1}},
        },
    ],
)
def test_schema_rejects_invalid_or_ambiguous_meaning(payload: object) -> None:
    with pytest.raises(ValidationError):
        DecisionRequest.model_validate(payload)


def test_compiler_preserves_option_ids_but_excludes_question_ids() -> None:
    request = _request()
    batch = compile_request(request, PROFILE)

    route = batch.questions[0]
    content = route.messages[0]["content"]
    assert route.id == "route"
    assert route.option_ids == ["support", "sales"]
    assert "A: support" in content
    assert 'B: sales - {"when":"new business"}' in content
    assert "route" not in content
    assert "email bounced" in content
    assert route.answer_prefix == "Answer:\n"
    assert route.labels == ["A", "B"]
    assert route.prompt_version == PROMPT_VERSION

    renamed = _request().model_copy(
        update={"questions": {"different-id": request.questions["route"]}}
    )
    assert compile_request(renamed, PROFILE).questions[0].messages == route.messages


def test_compiler_rejects_cardinality_above_backend_capacity() -> None:
    payload = {
        "state": "x",
        "questions": {
            "q": {
                "type": "choice",
                "instructions": "pick",
                "criteria": {str(index): None for index in range(5)},
            }
        },
    }
    with pytest.raises(ValueError, match="capacity"):
        compile_request(DecisionRequest.model_validate(payload), PROFILE)


def test_mapping_returns_typed_answers_and_literal_score_expectation() -> None:
    request = _request()
    batch = compile_request(request, PROFILE)
    worker = WorkerBatchResult.model_validate(
        {
            "type": "result",
            "id": "corr-1",
            "model_sha256": "a" * 64,
            "runtime_sha256": "b" * 64,
            "generated_tokens": 0,
            "callbacks_enabled": False,
            "execution_mode": "full",
            "questions": [
                {
                    "id": "route",
                    "label_logits": [0.0, math.log(3.0)],
                    "label_token_ids": [101, 102],
                    "allowed_label_mass": 0.25,
                    "full_vocabulary_argmax": {"token_id": 999, "logit": 8.0},
                    "prompt_sha256": "1" * 64,
                    "prompt_tokens": 19,
                    "processed_tokens": 19,
                    "reused_tokens": 0,
                    "cache_cleared": True,
                    "timing_ms": 2.5,
                },
                {
                    "id": "severity",
                    "label_logits": [math.log(0.2), math.log(0.3), math.log(0.5)],
                    "label_token_ids": [101, 102, 103],
                    "allowed_label_mass": 0.9,
                    "full_vocabulary_argmax": {"token_id": 103, "logit": 4.0},
                    "prompt_sha256": "2" * 64,
                    "prompt_tokens": 20,
                    "processed_tokens": 8,
                    "reused_tokens": 12,
                    "cache_cleared": False,
                    "timing_ms": 3.0,
                },
                {
                    "id": "urgent",
                    "label_logits": [math.log(0.8), math.log(0.2)],
                    "label_token_ids": [101, 102],
                    "allowed_label_mass": 0.7,
                    "full_vocabulary_argmax": {"token_id": 101, "logit": 3.0},
                    "prompt_sha256": "3" * 64,
                    "prompt_tokens": 18,
                    "processed_tokens": 6,
                    "reused_tokens": 12,
                    "cache_cleared": False,
                    "timing_ms": 2.0,
                },
            ],
        }
    )

    response = map_response(batch, worker, PROFILE, include_diagnostics=True)

    choice = response.answers["route"]
    assert choice.type == "choice"
    assert choice.choice == "sales"
    assert choice.probabilities == pytest.approx({"support": 0.25, "sales": 0.75})
    assert choice.confidence == pytest.approx(0.75)
    score = response.answers["severity"]
    assert score.type == "score"
    assert score.score == pytest.approx(1.3)
    assert score.legend == {"0": "low", "1": {"name": "medium"}, "2": ["high"]}
    assert score.confidence == pytest.approx(0.5)
    noul = response.answers["urgent"]
    assert noul.model_dump() == {"type": "noul", "noul": pytest.approx(0.8)}
    assert response.usage.model_dump() == {"input_tokens": 33, "output_tokens": 0}
    assert response.diagnostics is not None
    diagnostic = response.diagnostics["route"]
    assert diagnostic.token_mapping == {"A": 101, "B": 102}
    assert diagnostic.conditional_score_semantics == (
        "softmax_over_allowed_single_token_labels_v1"
    )
    assert diagnostic.prompt_version == PROMPT_VERSION
    assert (diagnostic.cache_cleared, diagnostic.reused_tokens) == (True, 0)
    reused = response.diagnostics["severity"]
    assert (reused.prompt_tokens, reused.processed_tokens) == (20, 8)
    assert (reused.cache_cleared, reused.reused_tokens) == (False, 12)

    first = worker.questions[0].model_copy(
        update={"processed_tokens": 7, "reused_tokens": 12, "cache_cleared": False}
    )
    carried = worker.model_copy(update={"questions": [first, *worker.questions[1:]]})
    with pytest.raises(ValueError, match="before this request"):
        map_response(batch, carried, PROFILE, include_diagnostics=False)


def test_worker_result_rejects_inconsistent_token_accounting() -> None:
    row = {
        "id": "q",
        "label_logits": [0.0, 1.0],
        "label_token_ids": [101, 102],
        "allowed_label_mass": 0.5,
        "full_vocabulary_argmax": {"token_id": 9, "logit": 2.0},
        "prompt_sha256": "1" * 64,
        "prompt_tokens": 20,
        "processed_tokens": 8,
        "reused_tokens": 12,
        "cache_cleared": False,
        "timing_ms": 1.0,
    }
    WorkerQuestionResult.model_validate(row)
    with pytest.raises(ValueError, match="cover the prompt"):
        WorkerQuestionResult.model_validate({**row, "processed_tokens": 20})
    with pytest.raises(ValueError, match="contradicts"):
        WorkerQuestionResult.model_validate({**row, "cache_cleared": True})


def test_compiler_marks_the_text_shared_by_every_question() -> None:
    batch = compile_request(_request(), PROFILE)
    heads = set()
    for question in batch.questions:
        content = question.messages[0]["content"].encode("utf-8")
        head = content[: question.shared_prefix_bytes].decode("utf-8")
        assert head.endswith("\nQUESTION:\n")
        assert question.worker_payload()["shared_prefix_bytes"] == len(
            head.encode("utf-8")
        )
        heads.add(head)
    assert len(heads) == 1


def test_mapping_rejects_partial_or_corrupt_worker_batches() -> None:
    request = _request()
    batch = compile_request(request, PROFILE)
    worker = WorkerBatchResult.model_validate(
        {
            "type": "result",
            "id": "corr-1",
            "model_sha256": "a" * 64,
            "runtime_sha256": "b" * 64,
            "generated_tokens": 0,
            "callbacks_enabled": False,
            "execution_mode": "full",
            "questions": [],
        }
    )
    with pytest.raises(ValueError, match="question count"):
        map_response(batch, worker, PROFILE, include_diagnostics=False)


def test_softmax_does_not_create_negative_tail_probability() -> None:
    probabilities = stable_softmax([0.0] * 6 + [-100.0])

    assert all(item >= 0.0 for item in probabilities)
    assert sum(probabilities) == pytest.approx(1.0)
    assert probabilities[-1] > 0.0
