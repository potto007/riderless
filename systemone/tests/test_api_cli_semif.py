"""CLI batching and SemIf interchange behavior."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from systemone.api.app import ApiConfig
from systemone.api.cli import evaluate_requests, load_requests, write_responses
from systemone.api.compiler import CompiledBatch
from systemone.api.schema import (
    BackendProfile,
    SystemOneRequest,
    WorkerBatchResult,
)
from systemone.api.semif import export_semif_row, import_semif_row

PROFILE = BackendProfile(
    model_id="local-gemma-systemone-v1",
    model_name="fixture",
    model_sha256="a" * 64,
    runtime_sha256="b" * 64,
    labels=["A", "B", "C"],
    label_token_ids=[1, 2, 3],
    context_size=2048,
    batch_size=256,
    ubatch_size=256,
    threads=8,
    max_questions=32,
    generated_tokens=0,
    callbacks_enabled=False,
    execution_mode="full",
)


class BatchBackend:
    def __init__(self) -> None:
        self.profile: BackendProfile | None = None
        self.ready = False
        self.started = 0
        self.closed = 0
        self.calls = 0

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
        self.calls += 1
        question = batch.questions[0]
        return WorkerBatchResult.model_validate(
            {
                "type": "result",
                "id": str(self.calls),
                "model_sha256": PROFILE.model_sha256,
                "runtime_sha256": PROFILE.runtime_sha256,
                "generated_tokens": 0,
                "callbacks_enabled": False,
                "execution_mode": "full",
                "questions": [
                    {
                        "id": question.id,
                        "label_logits": [math.log(0.25), math.log(0.75)],
                        "label_token_ids": [1, 2],
                        "allowed_label_mass": 0.8,
                        "full_vocabulary_argmax": {"token_id": 2, "logit": 2.0},
                        "prompt_sha256": "1" * 64,
                        "prompt_tokens": 12,
                        "processed_tokens": 12,
                        "reused_tokens": 0,
                        "cache_cleared": True,
                        "timing_ms": 1.0,
                    }
                ],
            }
        )


def _request(state: str) -> SystemOneRequest:
    return SystemOneRequest.model_validate(
        {
            "state": state,
            "questions": {
                "q": {
                    "type": "choice",
                    "instructions": "pick",
                    "criteria": {"left": None, "right": None},
                }
            },
        }
    )


def test_semif_roundtrip_preserves_row_id_state_and_option_order() -> None:
    row = {
        "id": "span-choice",
        "state": {"email": "Send it to billing@example.com"},
        "question": ["Which", "span"],
        "options": [
            {"id": "billing@example.com", "description": None},
            {"id": "sales@example.com", "description": "sales span"},
        ],
    }

    request = import_semif_row(row)
    exported = export_semif_row(request)

    assert list(request.questions) == ["span-choice"]
    question = request.questions["span-choice"]
    assert question.type == "choice"
    assert list(question.criteria) == ["billing@example.com", "sales@example.com"]
    assert exported.model_dump() == row


@pytest.mark.parametrize("count", [1, 17])
def test_semif_rejects_option_counts_outside_cli_profile(count: int) -> None:
    row = {
        "id": "q",
        "state": "x",
        "question": "pick",
        "options": [{"id": str(index), "description": None} for index in range(count)],
    }
    with pytest.raises(ValueError, match="2 to 16"):
        import_semif_row(row)


@pytest.mark.asyncio
async def test_cli_batch_loads_backend_once_and_writes_create_only_jsonl(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "requests.json"
    output_path = tmp_path / "responses.jsonl"
    input_path.write_text(
        """[
          {"state":"first","questions":{"q":{"type":"choice","instructions":"pick","criteria":{"left":null,"right":null}}}},
          {"state":"second","questions":{"q":{"type":"choice","instructions":"pick","criteria":{"left":null,"right":null}}}}
        ]"""
    )
    backend = BatchBackend()
    requests = load_requests(input_path)

    responses = await evaluate_requests(
        requests,
        ApiConfig(),
        backend_factory=lambda _: backend,
        diagnostics=True,
    )
    write_responses(output_path, responses)

    assert backend.started == 1 and backend.closed == 1 and backend.calls == 2
    assert [request.state for request in requests] == ["first", "second"]
    assert output_path.read_text().count("\n") == 2
    assert '"output_tokens":0' in output_path.read_text()
    with pytest.raises(FileExistsError):
        write_responses(output_path, responses)


def test_cli_loads_jsonl_and_rejects_nonfinite_json(tmp_path: Path) -> None:
    valid = _request("x").model_dump_json()
    path = tmp_path / "requests.jsonl"
    path.write_text(valid + "\n" + valid + "\n")
    assert len(load_requests(path)) == 2

    bad = tmp_path / "bad.json"
    bad.write_text(
        '{"state":{"x":NaN},"questions":{"q":{"type":"noul","instructions":"x"}}}'
    )
    with pytest.raises(ValueError, match="nonfinite"):
        load_requests(bad)
