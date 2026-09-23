"""Map validated worker branch results to typed public answers.

The scoring is identical to v1 (`stable_softmax` over allowed single-token
labels); only the accounting around it differs, so a decision that follows a
snapshot is typed the same way a v1 decision is.
"""

from __future__ import annotations

from riderless.api.mapping import stable_softmax
from riderless.api.schema import (
    Answer,
    ChoiceAnswer,
    NoulAnswer,
    ScoreAnswer,
)
from riderless.api.snapshots.compiler import CompiledSnapshotQuestion
from riderless.api.snapshots.errors import SnapshotProtocolError
from riderless.api.snapshots.schema import WorkerSnapshotQuestionResult


def map_answer(
    compiled: CompiledSnapshotQuestion,
    raw: WorkerSnapshotQuestionResult,
    expected_label_token_ids: list[int],
) -> Answer:
    if raw.id != compiled.id:
        raise SnapshotProtocolError("worker question ids or order differ from request")
    if raw.label_token_ids != expected_label_token_ids[: len(compiled.labels)]:
        raise SnapshotProtocolError("worker label token mapping differs from handshake")
    if len(raw.label_logits) != len(compiled.option_ids):
        raise SnapshotProtocolError("worker label count differs from request options")
    try:
        probabilities = stable_softmax(raw.label_logits)
    except ValueError as error:
        raise SnapshotProtocolError(
            "worker label logits cannot be normalized"
        ) from error
    mapped = dict(zip(compiled.option_ids, probabilities, strict=True))
    best = max(range(len(probabilities)), key=probabilities.__getitem__)
    confidence = max(probabilities)
    if compiled.kind == "choice":
        return ChoiceAnswer(
            choice=compiled.option_ids[best],
            probabilities=mapped,
            confidence=confidence,
        )
    if compiled.kind == "score":
        if compiled.legend is None:
            raise SnapshotProtocolError("compiled score is missing its legend")
        return ScoreAnswer(
            score=sum(index * value for index, value in enumerate(probabilities)),
            legend=compiled.legend,
            probabilities=mapped,
            confidence=confidence,
        )
    return NoulAnswer(noul=probabilities[0])
