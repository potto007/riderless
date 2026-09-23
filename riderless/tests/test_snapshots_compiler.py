"""Tests for the neutral context-profile compiler."""

from __future__ import annotations

import pytest

from riderless.api.schema import ChoiceQuestion, NoulQuestion
from riderless.api.snapshots.compiler import (
    CONTEXT_INSTRUCTION,
    CompileError,
    compile_context_followup,
    compile_context_prompt,
    compile_create,
    compile_readout_followup,
    compile_readout_prompt,
    content_bytes,
    state_prefix,
)
from riderless.api.snapshots.schema import (
    ContextInput,
    DecisionInput,
    PromptInput,
)

LABELS = ["A", "B", "C", "D"]


def _choice() -> ChoiceQuestion:
    return ChoiceQuestion(
        type="choice",
        instructions="What should the team investigate?",
        criteria={"new_request": "no refund yet", "missing_refund": "refund lost"},
    )


def test_state_prefix_and_content_bytes_are_exact() -> None:
    state = {"message": "refund not arrived"}
    expected = (
        f"{CONTEXT_INSTRUCTION}\n\nSTATE:\n"
        + '{"message":"refund not arrived"}'
        + "\n\n"
    )
    assert state_prefix(state) == expected
    assert content_bytes(state) == len(expected.encode("utf-8"))


def test_context_create_freezes_only_the_state_prefix() -> None:
    state = "the merchant issued a refund last week"
    compiled = compile_create(ContextInput(kind="context", state=state), LABELS)

    assert compiled.boundary == "context"
    assert compiled.answer_prefix == ""
    assert compiled.labels is None
    body = compiled.messages[0]["content"]
    assert body == state_prefix(state)
    assert compiled.content_bytes == len(body.encode("utf-8"))


def test_decision_create_appends_the_v1_question_block() -> None:
    state = "s"
    compiled = compile_create(
        DecisionInput(kind="decision", state=state, question=_choice()), LABELS
    )

    assert compiled.boundary == "readout"
    assert compiled.answer_prefix == "Answer:\n"
    assert compiled.labels == ["A", "B"]
    body = compiled.messages[0]["content"]
    # The state boundary is frozen before the continuation begins.
    assert body.startswith(state_prefix(state))
    block = body[len(state_prefix(state)) :]
    assert block.startswith("QUESTION:\n")
    assert "OPTIONS:" in block
    assert "A: new_request - no refund yet" in block
    assert "B: missing_refund - refund lost" in block
    assert block.endswith("Reply with one option label only.")
    # content_bytes still measures only the state prefix.
    assert compiled.content_bytes == len(state_prefix(state).encode("utf-8"))


def test_prompt_create_uses_the_prompt_heading() -> None:
    compiled = compile_create(
        PromptInput(kind="prompt", state="s", prompt="focus on the timeline"), LABELS
    )

    assert compiled.boundary == "readout"
    assert compiled.answer_prefix == ""
    body = compiled.messages[0]["content"]
    assert body == state_prefix("s") + "PROMPT:\nfocus on the timeline"


def test_context_followup_extends_the_frozen_prefix_verbatim() -> None:
    context = compile_create(ContextInput(kind="context", state="s"), LABELS)
    branch = compile_context_followup(context.messages, "status", _choice(), LABELS)

    content = branch.messages[0]["content"]
    assert content.startswith(context.messages[0]["content"])
    assert branch.answer_prefix == "Answer:\n"
    assert branch.option_ids == ["new_request", "missing_refund"]
    assert branch.labels == ["A", "B"]
    assert branch.kind == "choice"


def test_readout_followup_closes_the_stub_without_repeating_state() -> None:
    readout = compile_create(
        DecisionInput(kind="decision", state="s", question=_choice()), LABELS
    )
    branch = compile_readout_followup(
        readout.messages, readout.answer_prefix, "again", _choice(), LABELS
    )

    roles = [message["role"] for message in branch.messages]
    assert roles == ["user", "assistant", "user"]
    # The assistant stub carries the answer prefix with no trailing newline and
    # no selected label.
    assert branch.messages[1] == {"role": "assistant", "content": "Answer:"}
    # State is not repeated; the new user turn is just the question block.
    assert branch.messages[2]["content"].startswith("QUESTION:\n")
    assert "STATE:" not in branch.messages[2]["content"]


def test_prompt_branches_do_not_repeat_state_on_a_readout() -> None:
    context = compile_create(ContextInput(kind="context", state="s"), LABELS)
    context_branch = compile_context_prompt(context.messages, "why")
    assert context_branch.messages[0]["content"].startswith(
        context.messages[0]["content"]
    )
    assert context_branch.answer_prefix == ""

    readout = compile_create(
        DecisionInput(kind="decision", state="s", question=_choice()), LABELS
    )
    readout_branch = compile_readout_prompt(
        readout.messages, readout.answer_prefix, "why"
    )
    assert readout_branch.messages[-1] == {"role": "user", "content": "PROMPT:\nwhy"}
    assert readout_branch.messages[1]["role"] == "assistant"


def test_compiler_rejects_more_options_than_backend_labels() -> None:
    big = ChoiceQuestion(
        type="choice",
        instructions="pick",
        criteria={str(index): None for index in range(5)},
    )
    with pytest.raises(CompileError, match="capacity"):
        compile_create(DecisionInput(kind="decision", state="s", question=big), LABELS)


def test_noul_question_block_uses_true_false_labels() -> None:
    compiled = compile_create(
        DecisionInput(
            kind="decision",
            state="s",
            question=NoulQuestion(type="noul", instructions="urgent?"),
        ),
        LABELS,
    )
    block = compiled.messages[0]["content"][len(state_prefix("s")) :]
    assert "A: true" in block
    assert "B: false" in block
