"""Deterministic prompt compilation for the neutral context profile.

The `riderless-gemma-context-v1` profile freezes a reusable state boundary and
lets either a finite decision or a plain prompt continue from it. The question
block itself is the v1 block verbatim (reusing `_options`, `_question_heading`
and `render`), so a decision that follows a context boundary is scored exactly
as a v1 decision is. The Python side never handles tokens: it renders logical
messages and a byte boundary, and the worker owns tokenization and the prefix
check.
"""

from __future__ import annotations

from dataclasses import dataclass

from riderless.api.compiler import _options, _question_heading, render
from riderless.api.schema import Question, ScoreEntry
from riderless.api.snapshots.schema import (
    CONTEXT_PROMPT_VERSION,
    Boundary,
    DecisionInput,
    PromptInput,
    SnapshotInput,
)

CONTEXT_INSTRUCTION = (
    "The following state is provided as reusable context. Read it carefully. "
    "A question or prompt about this state may follow."
)
DECISION_ANSWER_PREFIX = "Answer:\n"
PROMPT_HEADING = "PROMPT:\n"


class CompileError(ValueError):
    """The request cannot be compiled against the active profile."""


Message = dict[str, str]


@dataclass(frozen=True, slots=True)
class CompiledCreate:
    """The create payload for one snapshot freeze point."""

    messages: list[Message]
    answer_prefix: str
    boundary: Boundary
    # The UTF-8 byte length of everything up to and including the STATE block.
    # The worker freezes the context prefix here; a readout ignores it.
    content_bytes: int
    # The label alphabet for a decision readout, so a 30 checkpoint can return
    # label logits at the answer position. None for a context or prompt freeze.
    labels: list[str] | None = None


@dataclass(frozen=True, slots=True)
class CompiledSnapshotQuestion:
    """One branch question, with what the mapper needs to type its answer."""

    id: str
    kind: str
    messages: list[Message]
    answer_prefix: str
    labels: list[str]
    option_ids: list[str]
    legend: dict[str, ScoreEntry] | None
    prompt_version: str = CONTEXT_PROMPT_VERSION

    def worker_payload(self, save_as: str | None) -> dict[str, object]:
        return {
            "id": self.id,
            "messages": self.messages,
            "answer_prefix": self.answer_prefix,
            "labels": self.labels,
            "prompt_version": self.prompt_version,
            "save_as": save_as,
        }


@dataclass(frozen=True, slots=True)
class CompiledPrompt:
    """A plain-prompt branch payload."""

    messages: list[Message]
    answer_prefix: str


def state_prefix(state: object) -> str:
    """The reusable context prefix: instruction, then the rendered state."""
    return f"{CONTEXT_INSTRUCTION}\n\nSTATE:\n{render(state)}\n\n"


def content_bytes(state: object) -> int:
    return len(state_prefix(state).encode("utf-8"))


def _labels_for(question: Question, labels: list[str]) -> list[str]:
    option_ids, _, _ = _options(question)
    if len(option_ids) > len(labels):
        raise CompileError(
            f"question has {len(option_ids)} options but backend capacity "
            f"is {len(labels)}"
        )
    return labels[: len(option_ids)]


def question_block(
    question: Question, labels: list[str]
) -> tuple[str, list[str], dict[str, ScoreEntry] | None]:
    """Render the v1 question block (from `QUESTION:` onward) for a continuation.

    Byte-for-byte the same block v1 emits, so the continuation is scored the
    same way; only its surrounding prefix differs.
    """
    chosen = _labels_for(question, labels)
    option_ids, descriptions, legend = _options(question)
    lines = ["QUESTION:", _question_heading(question)]
    instructions = render(question.instructions)
    if instructions:
        lines.extend(["", "INSTRUCTIONS:", instructions])
    lines.extend(["", "OPTIONS:"])
    for label, option_id, description in zip(
        chosen, option_ids, descriptions, strict=True
    ):
        rendered = render(description)
        line = f"{label}: {option_id}"
        lines.append(f"{line} - {rendered}" if rendered else line)
    lines.extend(["", "Reply with one option label only."])
    return "\n".join(lines), option_ids, legend


def compile_create(snapshot_input: SnapshotInput, labels: list[str]) -> CompiledCreate:
    """Build the create payload for a context, decision or prompt freeze."""
    prefix = state_prefix(snapshot_input.state)
    boundary_bytes = len(prefix.encode("utf-8"))
    if isinstance(snapshot_input, DecisionInput):
        block, _, _ = question_block(snapshot_input.question, labels)
        return CompiledCreate(
            messages=[{"role": "user", "content": prefix + block}],
            answer_prefix=DECISION_ANSWER_PREFIX,
            boundary="readout",
            content_bytes=boundary_bytes,
            labels=_labels_for(snapshot_input.question, labels),
        )
    if isinstance(snapshot_input, PromptInput):
        content = prefix + PROMPT_HEADING + snapshot_input.prompt
        return CompiledCreate(
            messages=[{"role": "user", "content": content}],
            answer_prefix="",
            boundary="readout",
            content_bytes=boundary_bytes,
        )
    # A context freeze has no continuation; the state prefix is the whole body.
    return CompiledCreate(
        messages=[{"role": "user", "content": prefix}],
        answer_prefix="",
        boundary="context",
        content_bytes=boundary_bytes,
    )


def _compiled_question(
    question_id: str,
    question: Question,
    messages: list[Message],
    labels: list[str],
) -> CompiledSnapshotQuestion:
    _, option_ids, legend = question_block(question, labels)
    return CompiledSnapshotQuestion(
        id=question_id,
        kind=question.type,
        messages=messages,
        answer_prefix=DECISION_ANSWER_PREFIX,
        labels=_labels_for(question, labels),
        option_ids=option_ids,
        legend=legend,
    )


def compile_context_followup(
    context_messages: list[Message],
    question_id: str,
    question: Question,
    labels: list[str],
) -> CompiledSnapshotQuestion:
    """Continue a context boundary with a new decision question.

    The parent's frozen user content is reused verbatim and the new block is
    appended, so the rendered branch is guaranteed to extend the frozen prefix.
    """
    if not context_messages:
        raise CompileError("context snapshot has no frozen messages")
    block, _, _ = question_block(question, labels)
    prefix = context_messages[0]["content"]
    messages = [{"role": "user", "content": prefix + block}]
    return _compiled_question(question_id, question, messages, labels)


def compile_readout_followup(
    readout_messages: list[Message],
    parent_answer_prefix: str,
    question_id: str,
    question: Question,
    labels: list[str],
) -> CompiledSnapshotQuestion:
    """Continue a readout snapshot with a new decision, without repeating state.

    The old turn is closed with an assistant stub carrying the answer prefix
    (no selected label), then a fresh user turn carries the new question block.
    """
    block, _, _ = question_block(question, labels)
    messages = [
        *readout_messages,
        {"role": "assistant", "content": parent_answer_prefix.rstrip("\n")},
        {"role": "user", "content": block},
    ]
    return _compiled_question(question_id, question, messages, labels)


def compile_context_prompt(
    context_messages: list[Message], prompt: str
) -> CompiledPrompt:
    if not context_messages:
        raise CompileError("context snapshot has no frozen messages")
    prefix = context_messages[0]["content"]
    return CompiledPrompt(
        messages=[{"role": "user", "content": prefix + PROMPT_HEADING + prompt}],
        answer_prefix="",
    )


def compile_readout_prompt(
    readout_messages: list[Message], parent_answer_prefix: str, prompt: str
) -> CompiledPrompt:
    messages = [
        *readout_messages,
        {"role": "assistant", "content": parent_answer_prefix.rstrip("\n")},
        {"role": "user", "content": PROMPT_HEADING + prompt},
    ]
    return CompiledPrompt(messages=messages, answer_prefix="")
