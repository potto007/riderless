"""Validate native results and map them to typed public answers."""

from __future__ import annotations

import math

from systemone.api.compiler import CompiledBatch
from systemone.api.schema import (
    Answer,
    BackendProfile,
    ChoiceAnswer,
    NoulAnswer,
    QuestionDiagnostics,
    ScoreAnswer,
    SystemOneResponse,
    Usage,
    WorkerBatchResult,
)


def stable_softmax(logits: list[float]) -> list[float]:
    if len(logits) < 2 or not all(math.isfinite(item) for item in logits):
        raise ValueError("label logits must be finite and contain at least two values")
    maximum = max(logits)
    weights = [math.exp(item - maximum) for item in logits]
    total = sum(weights)
    if not math.isfinite(total) or total <= 0:
        raise ValueError("label logits cannot be normalized")
    probabilities = [item / total for item in weights]
    if any(
        not math.isfinite(item) or item < 0 or item > 1 for item in probabilities
    ) or not math.isclose(sum(probabilities), 1.0, abs_tol=1e-10):
        raise ValueError("label probabilities are invalid")
    return probabilities


def map_response(
    batch: CompiledBatch,
    worker: WorkerBatchResult,
    profile: BackendProfile,
    *,
    include_diagnostics: bool,
) -> SystemOneResponse:
    if len(worker.questions) != len(batch.questions):
        raise ValueError("worker question count differs from request")
    if worker.model_sha256 != profile.model_sha256:
        raise ValueError("worker model provenance differs from handshake")
    if worker.runtime_sha256 != profile.runtime_sha256:
        raise ValueError("worker runtime provenance differs from handshake")
    if worker.generated_tokens != 0 or worker.callbacks_enabled:
        raise ValueError("worker violated the non-generation contract")
    if worker.execution_mode != "full":
        raise ValueError("worker did not use full-only execution")

    answers: dict[str, Answer] = {}
    diagnostics: dict[str, QuestionDiagnostics] = {}
    total_tokens = 0
    for compiled, raw in zip(batch.questions, worker.questions, strict=True):
        if raw.id != compiled.id:
            raise ValueError("worker question ids or order differ from request")
        expected_token_ids = profile.label_token_ids[: len(compiled.labels)]
        if raw.label_token_ids != expected_token_ids:
            raise ValueError("worker label token mapping differs from handshake")
        if len(raw.label_logits) != len(compiled.option_ids):
            raise ValueError("worker label count differs from request options")
        if raw is worker.questions[0] and raw.reused_tokens:
            raise ValueError("worker reused context from before this request")
        probabilities = stable_softmax(raw.label_logits)
        mapped = dict(zip(compiled.option_ids, probabilities, strict=True))
        best = max(range(len(probabilities)), key=probabilities.__getitem__)
        confidence = max(probabilities)
        if compiled.kind == "choice":
            answer: Answer = ChoiceAnswer(
                choice=compiled.option_ids[best],
                probabilities=mapped,
                confidence=confidence,
            )
        elif compiled.kind == "score":
            if compiled.legend is None:
                raise ValueError("compiled score is missing its legend")
            answer = ScoreAnswer(
                score=sum(index * value for index, value in enumerate(probabilities)),
                legend=compiled.legend,
                probabilities=mapped,
                confidence=confidence,
            )
        else:
            answer = NoulAnswer(noul=probabilities[0])
        answers[compiled.id] = answer
        total_tokens += raw.processed_tokens
        if include_diagnostics:
            diagnostics[compiled.id] = QuestionDiagnostics(
                raw_label_logits=raw.label_logits,
                token_mapping=dict(
                    zip(compiled.labels, raw.label_token_ids, strict=True)
                ),
                coverage=raw.allowed_label_mass,
                full_vocabulary_argmax=raw.full_vocabulary_argmax,
                prompt_sha256=raw.prompt_sha256,
                prompt_version=compiled.prompt_version,
                cache_cleared=raw.cache_cleared,
                prompt_tokens=raw.prompt_tokens,
                processed_tokens=raw.processed_tokens,
                reused_tokens=raw.reused_tokens,
                timing_ms=raw.timing_ms,
                model_sha256=worker.model_sha256,
                runtime_sha256=worker.runtime_sha256,
                execution_mode=worker.execution_mode,
                generated_tokens=worker.generated_tokens,
                callbacks_enabled=worker.callbacks_enabled,
            )
    if list(answers) != [question.id for question in batch.questions]:
        raise ValueError("worker response lost request answer ordering")
    return SystemOneResponse(
        model=profile.model_id,
        answers=answers,
        usage=Usage(input_tokens=total_tokens),
        diagnostics=diagnostics if include_diagnostics else None,
    )
