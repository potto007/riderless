"""ASGI contract tests using a deterministic lifecycle-aware backend."""

from __future__ import annotations

import asyncio
import json
import math
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
import pytest

from systemone.api.app import ApiConfig, create_app
from systemone.api.backend import BackendRequestError
from systemone.api.compiler import CompiledBatch
from systemone.api.schema import BackendProfile, WorkerBatchResult

PROFILE = BackendProfile(
    model_id="local-gemma-systemone-v1",
    model_name="fixture",
    model_sha256="a" * 64,
    runtime_sha256="b" * 64,
    labels=["A", "B", "C"],
    label_token_ids=[11, 12, 13],
    context_size=2048,
    batch_size=256,
    ubatch_size=256,
    threads=8,
    max_questions=32,
    generated_tokens=0,
    callbacks_enabled=False,
    execution_mode="full",
)


class DeterministicBackend:
    def __init__(self) -> None:
        self.profile: BackendProfile | None = None
        self.ready = False
        self.started = 0
        self.closed = 0
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.block = False
        self.corrupt_id = False
        self.failure: Exception | None = None

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
        if self.failure is not None:
            raise self.failure
        self.entered.set()
        if self.block:
            await self.release.wait()
        rows = []
        for index, question in enumerate(batch.questions):
            count = len(question.labels)
            if question.kind == "score" and count == 3:
                logits = [math.log(0.2), math.log(0.3), math.log(0.5)]
            else:
                logits = [float(item) for item in range(count)]
            rows.append(
                {
                    "id": "wrong" if self.corrupt_id else question.id,
                    "label_logits": logits,
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


class FailingStartupBackend(DeterministicBackend):
    async def start(self) -> BackendProfile:
        self.started += 1
        raise RuntimeError("fixture startup failure")


@asynccontextmanager
async def client_for(
    backend: DeterministicBackend, **overrides: Any
) -> AsyncIterator[httpx.AsyncClient]:
    config = ApiConfig(**overrides)
    app = create_app(config, backend_factory=lambda _: backend)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client,
    ):
        yield client


REQUEST = {
    "state": "payment duplicated",
    "questions": {
        "route": {
            "type": "choice",
            "instructions": "pick",
            "criteria": {"refund": None, "duplicate": None},
        },
        "severity": {
            "type": "score",
            "instructions": None,
            "criteria": ["low", "medium", "high"],
        },
        "true_check": {"type": "noul", "instructions": "A duplicate exists"},
    },
}


@pytest.mark.asyncio
async def test_routes_return_typed_results_profile_and_readiness() -> None:
    backend = DeterministicBackend()
    async with client_for(backend) as client:
        health = await client.get("/health")
        models = await client.get("/v1/models")
        response = await client.post("/v1/systemone?diagnostics=true", json=REQUEST)

    assert backend.started == 1 and backend.closed == 1
    assert health.json() == {
        "status": "ok",
        "model": "local-gemma-systemone-v1",
    }
    profile = models.json()["models"][0]
    assert profile["id"] == "local-gemma-systemone-v1"
    assert profile["limits"] == {
        "protocol_choice_options": 255,
        "backend_options": 3,
        "score_levels": 10,
        "context_tokens": 2048,
        "questions": 32,
        "request_bytes": 1048576,
        "concurrency": 1,
    }
    assert profile["capabilities"]["execution"] == "full_only"
    assert profile["capabilities"]["confidence"] == "max_probability_v1"
    assert profile["capabilities"]["score"] == "zero_based_expected_value"
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["answers"]["route"]["choice"] == "duplicate"
    assert body["answers"]["severity"]["score"] == pytest.approx(1.3)
    assert body["answers"]["true_check"] == {
        "type": "noul",
        "noul": pytest.approx(0.26894142137),
    }
    assert body["usage"] == {"input_tokens": 33, "output_tokens": 0}
    assert list(body["diagnostics"]) == ["route", "severity", "true_check"]


@pytest.mark.asyncio
async def test_request_and_transport_errors_are_structured() -> None:
    backend = DeterministicBackend()
    async with client_for(backend, max_request_bytes=200) as client:
        malformed = await client.post(
            "/v1/systemone",
            content=b"{",
            headers={"content-type": "application/json"},
        )
        media = await client.post(
            "/v1/systemone", content=b"x", headers={"content-type": "text/plain"}
        )
        invalid = await client.post(
            "/v1/systemone",
            json={"state": "x", "questions": {}},
        )
        unknown = await client.post(
            "/v1/systemone",
            json={
                "model": "some-other-model",
                "state": "x",
                "questions": {"q": {"type": "noul", "instructions": "x"}},
            },
        )
        oversized = await client.post(
            "/v1/systemone",
            content=b"{" + b" " * 201,
            headers={"content-type": "application/json"},
        )

    assert malformed.status_code == 400
    assert malformed.json()["error"]["code"] == "malformed_json"
    assert media.status_code == 415
    assert media.json()["error"]["retryable"] is False
    assert invalid.status_code == 422
    assert invalid.json()["error"]["field"] == "body.questions"
    assert unknown.status_code == 404
    assert unknown.json()["error"]["code"] == "unknown_model"
    assert oversized.status_code == 413


@pytest.mark.asyncio
async def test_concurrency_limit_rejects_busy_without_partial_work() -> None:
    backend = DeterministicBackend()
    backend.block = True
    async with client_for(backend) as client:
        first = asyncio.create_task(client.post("/v1/systemone", json=REQUEST))
        await backend.entered.wait()
        second = await client.post("/v1/systemone", json=REQUEST)
        backend.release.set()
        completed = await first

    assert second.status_code == 429
    assert second.json()["error"] == {
        "code": "busy",
        "message": "the model is serving another request",
        "retryable": True,
    }
    assert completed.status_code == 200


@pytest.mark.asyncio
async def test_startup_failure_still_closes_owned_backend() -> None:
    backend = FailingStartupBackend()
    app = create_app(ApiConfig(), backend_factory=lambda _: backend)

    with pytest.raises(RuntimeError, match="startup failure"):
        async with app.router.lifespan_context(app):
            pass

    assert backend.started == 1
    assert backend.closed == 1


@pytest.mark.asyncio
async def test_chunked_body_stops_reading_at_size_limit() -> None:
    backend = DeterministicBackend()

    async def chunks() -> AsyncIterator[bytes]:
        yield b"{" + b" " * 149
        yield b" " * 100
        raise AssertionError("body reader consumed beyond the size limit")

    async with client_for(backend, max_request_bytes=200) as client:
        response = await client.post(
            "/v1/systemone",
            content=chunks(),
            headers={"content-type": "application/json"},
        )

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "request_too_large"


@pytest.mark.asyncio
@pytest.mark.parametrize("corrupt", [False, True])
async def test_internal_failures_are_structured_and_corruption_closes_backend(
    corrupt: bool,
) -> None:
    backend = DeterministicBackend()
    backend.corrupt_id = corrupt
    if not corrupt:
        backend.failure = RuntimeError("private fixture detail")

    async with client_for(backend) as client:
        response = await client.post("/v1/systemone", json=REQUEST)
        health = await client.get("/health")

    assert response.status_code == 500
    assert response.json()["error"] == {
        "code": "internal_error",
        "message": "request failed internally",
        "retryable": False,
    }
    assert "private fixture detail" not in response.text
    if corrupt:
        assert health.status_code == 529


@pytest.mark.asyncio
async def test_control_token_content_is_a_distinct_client_error() -> None:
    backend = DeterministicBackend()
    backend.failure = BackendRequestError("preflight", reason="control_tokens")

    async with client_for(backend) as client:
        response = await client.post("/v1/systemone", json=REQUEST)
        health = await client.get("/health")

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "unsupported_content"
    assert health.status_code == 200


@pytest.mark.asyncio
async def test_media_type_matching_ignores_case() -> None:
    async with client_for(DeterministicBackend()) as client:
        response = await client.post(
            "/v1/systemone",
            content=json.dumps(REQUEST),
            headers={"content-type": "Application/JSON; charset=utf-8"},
        )

    assert response.status_code == 200
