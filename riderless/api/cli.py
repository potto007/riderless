"""Serve or batch the isolated non-generative API, and install its worker."""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import uvicorn

from riderless.api.app import (
    ApiConfig,
    ApiService,
    BackendFactory,
    _default_backend,
    create_app,
)
from riderless.api.native import fetch as worker_fetch
from riderless.api.schema import DecisionRequest, DecisionResponse


def _reject_constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON constant {value}")


def _loads(text: str) -> Any:
    return json.loads(text, parse_constant=_reject_constant)


def load_requests(path: Path) -> list[DecisionRequest]:
    try:
        if path.suffix.lower() == ".jsonl":
            raw = [
                _loads(line) for line in path.read_text().splitlines() if line.strip()
            ]
        else:
            value = _loads(path.read_text())
            raw = value if isinstance(value, list) else [value]
    except json.JSONDecodeError as error:
        raise ValueError("input contains malformed JSON") from error
    if not raw:
        raise ValueError("input contains no requests")
    return [DecisionRequest.model_validate(item) for item in raw]


async def evaluate_requests(
    requests: Sequence[DecisionRequest],
    config: ApiConfig,
    *,
    backend_factory: BackendFactory | None = None,
    diagnostics: bool = False,
) -> list[DecisionResponse]:
    factory = backend_factory or _default_backend
    service = ApiService(factory(config), config)
    responses: list[DecisionResponse] = []
    try:
        await service.start()
        for request in requests:
            responses.append(await service.evaluate(request, diagnostics=diagnostics))
        return responses
    finally:
        await service.close()


def write_responses(path: Path, responses: Sequence[DecisionResponse]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for response in responses:
            handle.write(response.model_dump_json(exclude_none=True) + "\n")


def _add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    defaults = ApiConfig()
    parser.add_argument("--model-path", type=Path, default=defaults.model_path)
    parser.add_argument("--worker", type=Path, default=defaults.worker_path)
    parser.add_argument("--manifest", type=Path, default=defaults.manifest_path)
    parser.add_argument(
        "--model-sha256",
        default=defaults.model_sha256,
        help="expected model hash; pass an empty string to skip the pin",
    )
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--context", type=int, default=defaults.context_size)
    parser.add_argument("--batch", type=int, default=defaults.batch_size)
    parser.add_argument("--ubatch", type=int, default=defaults.ubatch_size)
    parser.add_argument(
        "--batched",
        action="store_true",
        help="evaluate a request's questions in one batched decode (ADR 0004)",
    )
    parser.add_argument(
        "--batched-context",
        type=int,
        default=defaults.batched_context,
        help="KV cells a batched worker reserves, about 0.21 MiB each",
    )
    parser.add_argument("--threads", type=int, default=defaults.threads)
    parser.add_argument(
        "--request-timeout", type=float, default=defaults.request_timeout
    )
    parser.add_argument(
        "--startup-timeout", type=float, default=defaults.startup_timeout
    )


def _config(args: argparse.Namespace) -> ApiConfig:
    return ApiConfig(
        model_path=args.model_path,
        worker_path=args.worker,
        manifest_path=args.manifest,
        model_sha256=args.model_sha256 or None,
        gpu=args.gpu,
        context_size=args.context,
        batch_size=args.batch,
        ubatch_size=args.ubatch,
        batched=args.batched,
        batched_context=args.batched_context,
        threads=args.threads,
        request_timeout=args.request_timeout,
        startup_timeout=args.startup_timeout,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="evaluate JSON or JSONL into JSONL")
    _add_runtime_arguments(run)
    run.add_argument("--input", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--diagnostics", action="store_true")
    serve = commands.add_parser("serve", help="serve the ASGI application")
    _add_runtime_arguments(serve)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8090)
    worker = commands.add_parser("worker", help="manage the native worker install")
    worker_commands = worker.add_subparsers(dest="worker_command", required=True)
    fetch = worker_commands.add_parser(
        "fetch", help="download and verify a published worker bundle"
    )
    worker_fetch.add_arguments(fetch)
    args = parser.parse_args()
    if args.command == "worker":
        raise SystemExit(worker_fetch.run(fetch, args))
    config = _config(args)
    if args.command == "serve":
        uvicorn.run(create_app(config), host=args.host, port=args.port)
        return
    if args.output.exists():
        # Fail before the model load; open("x") below stays the real guard.
        parser.error("output exists; choose a new file")
    try:
        requests = load_requests(args.input)
        responses = asyncio.run(
            evaluate_requests(
                requests,
                config,
                diagnostics=args.diagnostics,
            )
        )
        write_responses(args.output, responses)
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
