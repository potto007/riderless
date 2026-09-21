"""Deterministic prompt compilation for runtime questions."""

from __future__ import annotations

import json
from dataclasses import dataclass

from systemone.api.schema import (
    BackendProfile,
    ChoiceQuestion,
    Entry,
    Question,
    ScoreEntry,
    ScoreQuestion,
    SystemOneRequest,
)

PROMPT_VERSION = "systemone-gemma-choice-v1"
ANSWER_PREFIX = "Answer:\n"
SYSTEM_INSTRUCTION = (
    "Use the supplied state to answer one finite decision question. "
    "Select exactly one listed label. Do not explain or produce other text."
)


@dataclass(frozen=True, slots=True)
class CompiledQuestion:
    id: str
    kind: str
    messages: list[dict[str, str]]
    answer_prefix: str
    labels: list[str]
    option_ids: list[str]
    legend: dict[str, ScoreEntry] | None
    shared_prefix_bytes: int = 0
    prompt_version: str = PROMPT_VERSION

    def worker_payload(self) -> dict[str, object]:
        return {
            "id": self.id,
            "messages": self.messages,
            "answer_prefix": self.answer_prefix,
            "labels": self.labels,
            "prompt_version": self.prompt_version,
            "shared_prefix_bytes": self.shared_prefix_bytes,
        }


@dataclass(frozen=True, slots=True)
class CompiledBatch:
    model_id: str
    questions: list[CompiledQuestion]


def render(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _options(
    question: Question,
) -> tuple[list[str], list[Entry], dict[str, ScoreEntry] | None]:
    if isinstance(question, ChoiceQuestion):
        return list(question.criteria), list(question.criteria.values()), None
    if isinstance(question, ScoreQuestion):
        ids = [str(index) for index in range(len(question.criteria))]
        legend: dict[str, ScoreEntry] = dict(zip(ids, question.criteria, strict=True))
        return ids, list(question.criteria), legend
    rubric = question.criteria
    descriptions: list[Entry] = [
        rubric.true if rubric is not None else None,
        rubric.false if rubric is not None else None,
    ]
    return ["true", "false"], descriptions, None


def _question_heading(question: Question) -> str:
    if isinstance(question, ChoiceQuestion):
        return "Choose the single best option for the state."
    if isinstance(question, ScoreQuestion):
        return "Choose one ordered level, from lowest to highest."
    return "Decide whether the stated condition is true or false."


def _compile_question(
    question_id: str,
    question: Question,
    state: object,
    profile: BackendProfile,
    share_prefix: bool,
) -> CompiledQuestion:
    option_ids, descriptions, legend = _options(question)
    if len(option_ids) > len(profile.labels):
        raise ValueError(
            f"question {question_id!r} has {len(option_ids)} options but backend "
            f"capacity is {len(profile.labels)}"
        )
    labels = profile.labels[: len(option_ids)]
    shared = [SYSTEM_INSTRUCTION, "", "STATE:", render(state), "", "QUESTION:"]
    lines = [*shared, _question_heading(question)]
    instructions = render(question.instructions)
    if instructions:
        lines.extend(["", "INSTRUCTIONS:", instructions])
    lines.extend(["", "OPTIONS:"])
    for label, option_id, description in zip(
        labels, option_ids, descriptions, strict=True
    ):
        rendered = render(description)
        line = f"{label}: {option_id}"
        lines.append(f"{line} - {rendered}" if rendered else line)
    lines.extend(["", "Reply with one option label only."])
    return CompiledQuestion(
        id=question_id,
        kind=question.type,
        messages=[{"role": "user", "content": "\n".join(lines)}],
        answer_prefix=ANSWER_PREFIX,
        labels=labels,
        option_ids=option_ids,
        legend=legend,
        # Every question of a request starts with this text, so the worker may
        # prefill it once. It is a hint only; the prompt itself is unchanged.
        shared_prefix_bytes=(
            len(("\n".join(shared) + "\n").encode("utf-8")) if share_prefix else 0
        ),
    )


def compile_request(
    request: SystemOneRequest, profile: BackendProfile, *, share_prefix: bool = True
) -> CompiledBatch:
    if request.model != profile.model_id:
        raise ValueError(f"unknown model {request.model!r}")
    if len(request.questions) > profile.max_questions:
        raise ValueError("request exceeds backend question capacity")
    return CompiledBatch(
        model_id=profile.model_id,
        questions=[
            _compile_question(
                question_id, question, request.state, profile, share_prefix
            )
            for question_id, question in request.questions.items()
        ],
    )
