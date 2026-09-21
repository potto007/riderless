"""Distinct ASGI application for the strict non-generative API."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from riderless.api.backend import (
    Backend,
    BackendExecutionError,
    BackendProtocolError,
    BackendRequestError,
    BackendUnavailableError,
)
from riderless.api.compiler import compile_request
from riderless.api.mapping import map_response
from riderless.api.schema import (
    MODEL_ID,
    BackendProfile,
    DecisionRequest,
    DecisionResponse,
    ErrorBody,
    ErrorResponse,
)

LOGGER = logging.getLogger("riderless.api.app")
BackendFactory = Callable[["ApiConfig"], Backend]
DECISION_PATH = "/v1/decisions"
# Compatibility alias. Same handler, same bodies; an interoperability path for
# clients written against this project's earlier request shape.
COMPATIBILITY_PATH = "/v1/systemone"
DECISION_PATHS = frozenset({DECISION_PATH, COMPATIBILITY_PATH})


@dataclass(frozen=True, slots=True)
class ApiConfig:
    # Relative defaults for a checkout-local layout. Download the GGUF yourself
    # and either place it here or pass an explicit path.
    model_path: Path = Path("models/gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf")
    worker_path: Path = Path("build/api-worker/build/riderless-worker")
    manifest_path: Path = Path("build/api-worker/build.json")
    # Optional model pin. Set it to your GGUF's sha256 and startup refuses any
    # other file. None accepts whatever `model_path` points at.
    model_sha256: str | None = None
    gpu: bool = False
    context_size: int = 2048
    batch_size: int = 256
    ubatch_size: int = 256
    # False makes every question prefill its whole prompt from an empty context.
    share_prefix: bool = True
    # Opt-in batched evaluation (ADR 0004). It is not bit-identical to the
    # default, and a question's result then depends on its siblings.
    batched: bool = False
    # KV cells a batched worker reserves: the shared prefix plus every
    # remainder. A request that needs more falls back to sequential. Each cell
    # costs about 0.21 MiB of KV on this model (1,760 MiB measured at 8,192
    # cells), so the cap below keeps a typo from reserving the whole card.
    batched_context: int = 8192
    threads: int = 8
    max_questions: int = 32
    max_request_bytes: int = 1024 * 1024
    max_response_bytes: int = 4 * 1024 * 1024
    request_timeout: float = 120.0
    startup_timeout: float = 600.0

    def __post_init__(self) -> None:
        positive = {
            "context_size": self.context_size,
            "batch_size": self.batch_size,
            "ubatch_size": self.ubatch_size,
            "threads": self.threads,
            "max_questions": self.max_questions,
            "max_request_bytes": self.max_request_bytes,
            "max_response_bytes": self.max_response_bytes,
            "batched_context": self.batched_context,
        }
        if any(value <= 0 for value in positive.values()):
            raise ValueError("API limits must be positive")
        if self.max_questions > 32:
            raise ValueError("max_questions cannot exceed 32")
        # Both bounds describe the batched cache, so neither applies to a
        # sequential worker, which never allocates one and leaves the default
        # batched_context unused.
        if self.batched:
            if self.batched_context < self.context_size:
                raise ValueError("batched_context cannot be below context_size")
            if self.batched_context > self.max_questions * self.context_size:
                raise ValueError(
                    "batched_context cannot exceed max_questions * context_size"
                )
        if self.request_timeout <= 0 or self.startup_timeout <= 0:
            raise ValueError("timeouts must be positive")


class BusyError(RuntimeError):
    """The one-request backend is occupied."""


class ApiService:
    def __init__(self, backend: Backend, config: ApiConfig) -> None:
        self.backend = backend
        self.config = config
        self.profile: BackendProfile | None = None
        self._guard = asyncio.Lock()
        self._busy = False

    async def start(self) -> None:
        self.profile = await asyncio.wait_for(
            self.backend.start(), timeout=self.config.startup_timeout
        )

    async def close(self) -> None:
        await self.backend.close()

    async def evaluate(
        self, request: DecisionRequest, *, diagnostics: bool
    ) -> DecisionResponse:
        if self.profile is None or not self.backend.ready:
            raise BackendUnavailableError("backend unavailable")
        async with self._guard:
            if self._busy:
                raise BusyError("backend busy")
            self._busy = True
        try:
            batch = compile_request(
                request, self.profile, share_prefix=self.config.share_prefix
            )
            worker = await asyncio.wait_for(
                self.backend.evaluate(batch, timeout=self.config.request_timeout),
                timeout=self.config.request_timeout,
            )
            try:
                return map_response(
                    batch, worker, self.profile, include_diagnostics=diagnostics
                )
            except ValueError as error:
                await self.backend.close()
                raise BackendProtocolError(
                    "backend response differs from the request"
                ) from error
        finally:
            async with self._guard:
                self._busy = False


def _error(
    status: int,
    code: str,
    message: str,
    *,
    field: str | None = None,
    retryable: bool = False,
) -> JSONResponse:
    body = ErrorResponse(
        error=ErrorBody(
            code=code,
            message=message,
            field=field,
            retryable=retryable,
        )
    )
    return JSONResponse(status_code=status, content=body.model_dump(exclude_none=True))


def _default_backend(config: ApiConfig) -> Backend:
    from riderless.api.native_backend import NativeBackend

    return NativeBackend(
        model_path=config.model_path,
        worker_path=config.worker_path,
        manifest_path=config.manifest_path,
        gpu=config.gpu,
        context_size=config.context_size,
        batch_size=config.batch_size,
        ubatch_size=config.ubatch_size,
        batched=config.batched,
        batched_context=config.batched_context,
        threads=config.threads,
        max_questions=config.max_questions,
        max_response_bytes=config.max_response_bytes,
        startup_timeout=config.startup_timeout,
        expected_model_sha256=config.model_sha256,
    )


def create_app(
    config: ApiConfig | None = None,
    *,
    backend_factory: BackendFactory | None = None,
) -> FastAPI:
    """Create an app without starting a model or touching the GPU."""
    resolved = config or ApiConfig()
    factory = backend_factory or _default_backend

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        backend = factory(resolved)
        service = ApiService(backend, resolved)
        app.state.service = service
        try:
            await service.start()
            yield
        finally:
            await service.close()

    app = FastAPI(
        title="Riderless Decision API",
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def request_bounds(request: Request, call_next: Any) -> Any:
        if request.url.path in DECISION_PATHS and request.method == "POST":
            content_type = request.headers.get("content-type", "")
            media = content_type.split(";", 1)[0].strip().lower()
            if media != "application/json":
                return _error(
                    415,
                    "unsupported_media_type",
                    "content-type must be application/json",
                )
            length = request.headers.get("content-length")
            if length is not None:
                try:
                    if int(length) > resolved.max_request_bytes:
                        return _error(
                            413, "request_too_large", "request body is too large"
                        )
                except ValueError:
                    return _error(
                        400, "invalid_content_length", "invalid content-length"
                    )
            chunks: list[bytes] = []
            received = 0
            async for chunk in request.stream():
                received += len(chunk)
                if received > resolved.max_request_bytes:
                    return _error(413, "request_too_large", "request body is too large")
                chunks.append(chunk)
            request._body = b"".join(chunks)
        return await call_next(request)

    @app.exception_handler(RequestValidationError)
    async def validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        del request
        errors = exc.errors()
        malformed = any(item.get("type") == "json_invalid" for item in errors)
        if malformed:
            return _error(400, "malformed_json", "request body is not valid JSON")
        first = errors[0] if errors else {}
        location = first.get("loc", ())
        field = ".".join(str(part) for part in location) or None
        return _error(422, "invalid_request", "request schema is invalid", field=field)

    @app.get("/health")
    async def health(request: Request) -> JSONResponse:
        service: ApiService = request.app.state.service
        if service.profile is None or not service.backend.ready:
            return _error(
                529,
                "unavailable",
                "model backend is unavailable",
                retryable=True,
            )
        return JSONResponse({"status": "ok", "model": MODEL_ID})

    @app.get("/v1/models")
    async def models(request: Request) -> dict[str, object]:
        service: ApiService = request.app.state.service
        profile = service.profile
        if profile is None:
            return {"models": []}
        return {
            "models": [
                {
                    "id": profile.model_id,
                    "limits": {
                        "protocol_choice_options": 255,
                        "backend_options": len(profile.labels),
                        "score_levels": 10,
                        "context_tokens": profile.context_size,
                        "questions": min(profile.max_questions, resolved.max_questions),
                        "request_bytes": resolved.max_request_bytes,
                        "concurrency": 1,
                    },
                    "capabilities": {
                        "primitives": ["choice", "score", "noul"],
                        "execution": "full_only",
                        "generation": False,
                        "confidence": "max_probability_v1",
                        "score": "zero_based_expected_value",
                        "probabilities": ("conditional_on_allowed_single_token_labels"),
                    },
                    "runtime": {
                        "context": profile.context_size,
                        "batch": profile.batch_size,
                        "ubatch": profile.ubatch_size,
                        "threads": profile.threads,
                        "batched": profile.batched_mode,
                        "batched_context": profile.batched_context,
                        "generated_tokens": profile.generated_tokens,
                        "callbacks_enabled": profile.callbacks_enabled,
                        # False means this build links a llama.cpp revision
                        # other than the one the published numbers came from.
                        "llama_revision": profile.llama_revision,
                        "tested_revision": profile.tested_revision,
                    },
                }
            ]
        }

    async def decide(
        request: DecisionRequest,
        raw_request: Request,
        diagnostics: bool = Query(default=False),
    ) -> DecisionResponse | JSONResponse:
        if request.model != MODEL_ID:
            return _error(404, "unknown_model", "requested model is not available")
        service: ApiService = raw_request.app.state.service
        try:
            return await service.evaluate(request, diagnostics=diagnostics)
        except BusyError:
            return _error(
                429,
                "busy",
                "the model is serving another request",
                retryable=True,
            )
        except TimeoutError:
            return _error(408, "timeout", "request timed out", retryable=True)
        except BackendUnavailableError:
            return _error(
                529,
                "unavailable",
                "model backend is unavailable",
                retryable=True,
            )
        except (BackendExecutionError, BackendProtocolError):
            return _error(500, "internal_error", "request failed internally")
        except BackendRequestError as error:
            if error.reason == "control_tokens":
                return _error(
                    422,
                    "unsupported_content",
                    "state or question text contains model control tokens",
                )
            return _error(422, "budget_error", "request exceeds model limits")
        except ValueError:
            return _error(422, "budget_error", "request exceeds model limits")
        except Exception:
            LOGGER.exception("unexpected failure while serving a request")
            return _error(500, "internal_error", "request failed internally")

    app.add_api_route(
        DECISION_PATH,
        decide,
        methods=["POST"],
        response_model=DecisionResponse,
        response_model_exclude_none=True,
        name="decisions",
        summary="Answer a batch of typed questions",
    )
    # One handler, two paths, so the alias can never drift from the primary.
    app.add_api_route(
        COMPATIBILITY_PATH,
        decide,
        methods=["POST"],
        response_model=DecisionResponse,
        response_model_exclude_none=True,
        name="decisions_compatibility_alias",
        summary=f"Compatibility alias for POST {DECISION_PATH}",
    )

    return app


__all__ = ["ApiConfig", "create_app"]
