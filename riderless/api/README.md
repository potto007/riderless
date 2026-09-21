# Riderless Decision API v1

This package provides the isolated, full-depth non-generative API. It loads one
Gemma model in one owned native child and reads logits for a validated finite
label alphabet after prefill. It does not create a sampler, append an answer
token, or feed an answer token back into the model.

The API model id is `local-gemma-riderless-v1`. It names this local
deployment only and claims no parity with any hosted service.

## Build the native worker

The build is create-only and verifies the base runtime before compiling:

```bash
# 1. Produce the llama.cpp base runtime (headers + shared libraries). The
#    default revision is the tested release v0.4.1; --revision takes any other
#    tag or commit. --cuda-architectures keeps a CUDA build to your own GPU.
python scripts/riderless/build_base_runtime.py \
  --out build/llama-base --cuda --cuda-architectures 120

# 2. Compile the worker against it.
python -m riderless.api.native.build \
  --base build/llama-base \
  --output build/api-worker
```

The result is `build/api-worker/build/riderless-worker`. Its sibling
`build.json` records the worker source, executable, bundled sha256 helper,
linked runtime, and llama.cpp revision hashes. Startup enforces the executable,
runtime files, and runtime bundle hashes; the source and helper hashes are a
build record and are not rechecked at startup.

The llama.cpp revision is provenance, not a gate. Every hash above is enforced
whatever the revision, and a revision other than the tested one only logs a
warning that the published measurements were taken on the tested revision and
may not reproduce. `GET /v1/models` reports it as `runtime.llama_revision` and
`runtime.tested_revision`. The model is pinned separately by
`ApiConfig.model_sha256` (CLI `--model-sha256`): set it to your GGUF's sha256
and startup refuses any other file. Building and running the CPU helper test
does not load the model or start GPU work.

## Create the ASGI app

Importing the package or creating the app does not start a child process. The
ASGI lifespan owns startup and shutdown:

```python
from pathlib import Path

from riderless.api.app import ApiConfig, create_app

app = create_app(
    ApiConfig(
        model_path=Path("models/gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf"),
        worker_path=Path("build/api-worker/build/riderless-worker"),
        manifest_path=Path("build/api-worker/build.json"),
        gpu=True,
    )
)
```

`ApiConfig` defaults are context 2048, batch and ubatch 256, 8 threads, at
most 32 questions, a 1 MiB HTTP request, a 4 MiB worker response, a 120 second
request timeout, and a 600 second startup timeout. `gpu` defaults to `False` and
must be opted into explicitly.

Run a loopback server with the same factory:

```bash
python -m riderless.api.cli serve \
  --worker build/api-worker/build/riderless-worker \
  --manifest build/api-worker/build.json \
  --model-path models/gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf \
  --gpu --host 127.0.0.1 --port 8090
```

`serve` binds a loopback port and runs in the foreground; put it behind a
process supervisor before any standing use. Validation and batch work need no
port at all: use the ASGI harness or `run` below. With `--gpu` and the default
2048-token context the worker holds about 18 GiB of device memory (roughly
16 GiB of weights plus KV and compute buffers), so confirm the card has that
much free before starting it.

The supported routes are `POST /v1/decisions`, `GET /v1/models`, and
`GET /health`. FastAPI also exposes machine-readable OpenAPI at `/openapi.json`.
`POST /v1/systemone` is a compatibility alias for `POST /v1/decisions`, offered
as an interoperability path for clients written against that request shape.

## HTTP request and result

```json
{
  "model": "local-gemma-riderless-v1",
  "state": {"message": "The card was charged twice"},
  "questions": {
    "route": {
      "type": "choice",
      "instructions": "Choose the best route",
      "criteria": {
        "billing": null,
        "technical": "A software defect"
      }
    },
    "severity": {
      "type": "score",
      "instructions": null,
      "criteria": ["low", "medium", "high"]
    },
    "duplicate": {
      "type": "noul",
      "instructions": "The state describes a duplicate charge"
    }
  }
}
```

For label probabilities `[0.2, 0.3, 0.5]`, Score returns the zero-based
expected value `1.3`. Noul returns only `P(true)`:

```json
{
  "model": "local-gemma-riderless-v1",
  "answers": {
    "route": {
      "type": "choice",
      "choice": "billing",
      "probabilities": {"billing": 0.8, "technical": 0.2},
      "confidence": 0.8
    },
    "severity": {
      "type": "score",
      "score": 1.3,
      "legend": {"0": "low", "1": "medium", "2": "high"},
      "probabilities": {"0": 0.2, "1": 0.3, "2": 0.5},
      "confidence": 0.5
    },
    "duplicate": {"type": "noul", "noul": 0.9}
  },
  "usage": {"input_tokens": 384, "output_tokens": 0}
}
```

Use `?diagnostics=true` to add a top-level map keyed by question id. Each entry
contains `raw_label_logits`, `token_mapping`, `coverage`,
`full_vocabulary_argmax`, `conditional_score_semantics`, `confidence_formula`,
`prompt_sha256`, `prompt_version`, `cache_cleared`, `prompt_tokens`,
`processed_tokens`, `reused_tokens`, `timing_ms`, `model_sha256`, `runtime_sha256`, `execution_mode`,
`generated_tokens`, and `callbacks_enabled`.

Native stderr is never copied into HTTP errors. Operators can enable the
`riderless.api.native` logger at DEBUG or inspect the bounded
`app.state.service.backend.stderr_tail`. The exact owned child is available as
`app.state.service.backend.pid`, and its validated handshake as `.profile`.

## Batch CLI

The batch CLI accepts one request object, an array of request objects, or JSONL.
It starts one backend for the complete batch and creates the output JSONL only
when the path does not already exist:

```bash
python -m riderless.api.cli run \
  --input requests.jsonl --output results.jsonl \
  --worker build/api-worker/build/riderless-worker \
  --manifest build/api-worker/build.json \
  --model-path models/gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf \
  --gpu --diagnostics
```

## SemIf interchange

A separate adapter reads and writes SemIf-style JSONL rows: `id`, `state`,
`question`, and 2 to 16 ordered `{id, description}` options. One row maps to one
Choice question and keeps the row id and option order. SemIf is an unrelated
open-source project; this is an interchange convenience, not an integration.

```bash
python -m riderless.api.semif to-api \
  --input semif.jsonl --output requests.jsonl
python -m riderless.api.semif from-api \
  --input requests.jsonl --output semif-roundtrip.jsonl
```

## Limits and semantics

`GET /v1/models` reports the label capacity validated from the loaded
tokenizer. Choice accepts 2 to 255 options at the wire boundary, then rejects a
question above that backend capacity before inference. Score accepts 2 to 10
levels. Noul requires instructions or a true/false rubric. Option ids and order
are preserved; question ids are correlation metadata and are absent from model
prompts.

Each question is compiled into its own prompt from the state plus that
question's own instructions and criteria. It never sees sibling questions, so a
policy or definition that a question depends on must be in the state or repeated
in that question's instructions.

Within one request, the text every question shares (the system instruction and
the state) is prefilled once and reused: the first question clears the context
and prefills its whole prompt, and each later question trims the context back to
the shared prefix and prefills only its own remainder. The split point depends
only on a question's own prompt, and the prefix is always prefilled in the same
chunks, so a question gets the same logits alone, reordered, or beside any
siblings. Nothing is kept between requests. Diagnostics report the accounting
per question: `processed_tokens + reused_tokens == prompt_tokens`, and
`cache_cleared` is true exactly when nothing was reused. `usage.input_tokens`
counts the tokens actually prefilled. Parallel GPU requests are outside v1. The
service permits one
active inference request and immediately returns 429 while busy. Conditional
label probabilities are finite and normalized, but they are not calibrated
correctness probabilities. Choice and Score confidence is
`max_probability_v1`.

State, instructions, and option text must not spell a model control token such
as `<turn|>` or `<|turn>`. The worker parses special tokens so the chat template works,
so such text could forge a turn boundary. It is rejected before inference with
422 `unsupported_content`. Ordinary markup and angle brackets are accepted.

Errors use `{error: {code, message, field?, retryable}}`. Statuses distinguish
malformed input (400), unknown model (404), timeout (408), oversized body
(413), unsupported media (415), schema or model budget failures (422), busy
(429), unavailable backend (529), and internal failure (500). A batch never
returns a successful partial result.
