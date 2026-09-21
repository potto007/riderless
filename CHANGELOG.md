# Changelog

Notable changes to riderless. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
intends to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
once it tags a release.

The project has no tagged release yet, so everything lives under Unreleased.
Until the first tag, the HTTP contract, the CLI flags, and the native worker
protocol can change without a deprecation period.

## Unreleased

### Added

- Opt-in batched question evaluation. `ApiConfig.batched` (CLI, runner and
  worker `--batched`) gives every question of a request its own KV sequence over
  one copy of the shared prefix and decodes all the remainders together.
  `ApiConfig.batched_context` (worker `--batched-context`, default 8192 cells)
  sizes the unified KV cache and may not exceed `max_questions * context_size`.
  The default stays sequential and reproduces the recorded v0.4.1 suite run bit
  for bit.
- A batched worker names the regime that answered each request in the public
  response body as `evaluation: {mode, fallback}`, and a request that needs more
  KV cells than `batched_context` is answered by the sequential path with
  `fallback: "context"` rather than being truncated or refused. Diagnostics gain
  `evaluation_mode` and `batch_sequences` per question. `GET /v1/models` reports
  `runtime.batched` and `runtime.batched_context`.
- `scripts/riderless/compare_observations.py` and
  `scripts/riderless/compare_fallback.py`, which reduce two recorded runs to
  answer changes, probability and raw logit moves, tokens and latency.
- [ADR 0004](docs/decisions/0004-optional-batched-question-evaluation.md) and
  the [batched-mode results](docs/results/batched-mode.md).

### Changed

- Worker protocol version 3. The handshake carries `batched_mode` and
  `batched_context`, and a result carries `batched_fallback` plus the two new
  per-question fields. The backend refuses a version 2 worker.
- The validation harness stops asserting exact sibling independence in batched
  mode and records the solo, reversed and renamed deltas instead. A repeat of an
  identical request must still differ by exactly 0.0 in both modes. It also
  gained a cross-question contamination probe, run in both question orders, that
  fails the run if an adversarial sibling moves the target further than a
  neutral sibling of the same length, and a request sized to force the
  sequential fallback.

- The llama.cpp revision is a tested default instead of a hard pin. The build
  records llama.cpp release `v0.4.1` (commit
  `b29c606e28a01b1bc8c1351026a0fa6e616bf6c4`) as `TESTED_LLAMA_TAG` and
  `TESTED_LLAMA_REVISION`, and `build_base_runtime.py` fetches that tag by
  default. Asking for the tested tag verifies that it still resolves to the
  recorded commit, so a moved tag fails the build.
- `build_base_runtime.py --revision <tag-or-sha>` builds any other revision.
  The base `build.json` now records `llama_ref`, the resolved `llama_revision`,
  and `tested_revision`.
- Startup no longer refuses a build made from another llama.cpp revision. Every
  hash check is unchanged, and an untested revision logs one warning saying that
  the published measurements were taken on the tested revision and may not
  reproduce. Previously such a build was rejected outright.
- `GET /v1/models` reports `runtime.llama_revision` and
  `runtime.tested_revision`; `BackendProfile` carries the same two fields.
- The worker is ported to the v0.4.1 common library API: chat messages are
  handed to `common_chat_msgs_parse_oaicompat` as `common_json`, and the
  bundled sha256 helper is located under either `vendor/hash` or the older
  `examples/gguf-hash/deps`.
- `build.json` renames `frozen_helper_sha256` to `helper_sha256`.
- CI uses `astral-sh/setup-uv@v10.2.0`, which runs on Node 24. v6 ran on the
  deprecated Node 20.

### Added

- `build_base_runtime.py --cuda-architectures`, passed through as
  `CMAKE_CUDA_ARCHITECTURES`. Omitted, llama.cpp's own default applies.
- [ADR 0003](docs/decisions/0003-tested-default-revision-instead-of-a-hard-pin.md)
  and the [v0.4.1 re-validation results](docs/results/llama-v0.4.1-revalidation.md),
  which state which published numbers were measured on which revision.

- `riderless` Python package: a local, non-generative decision API. One owned
  llama.cpp child process prefills a compiled prompt and reads final-position
  logits over single-token labels, so a request produces zero generated
  tokens.
- Three caller-defined question types, batched in one request: `choice`
  (pick one of a caller-supplied label set), `score` (a bounded numeric
  rating), and `noul` (true or false against caller criteria).
- HTTP surface: `POST /v1/decisions`, `GET /v1/models`, and `GET /health`,
  served by a FastAPI application whose lifespan owns the worker child.
  Importing the package or building the app starts no process.
  `POST /v1/systemone` is a compatibility alias for `POST /v1/decisions`,
  offered as an interoperability path for clients written against that
  request shape.
- `riderless.api.cli` with `run` (evaluate a JSON or JSONL file of requests
  into JSONL, no port needed) and `serve` (run the ASGI app on a loopback
  listener).
- SemIf row import and export helpers in `riderless.api.semif`.
- Native worker under `riderless/api/native`: a C++ worker built against a
  user-provided llama.cpp checkout, plus `riderless.api.native.build`, which
  verifies the base runtime before compiling and records source, executable,
  runtime, and llama.cpp revision hashes in a build manifest. Startup pins the
  worker executable and the runtime files and bundle; the model file's SHA-256
  is always computed and is enforced when `--model-sha256` is configured.
- Validation and use-case scripts under `scripts/riderless`, with the
  matching request and expectation fixtures under `riderless/examples`.
- Architecture decision records under `docs/decisions` and measured results
  under `docs/results`.
- Community and governance files: Apache-2.0 license, NOTICE, contribution
  guide with DCO sign-off, Code of Conduct, security policy, issue and pull
  request templates, and a CPU-only CI workflow.

### Notes

- Model weights are not distributed with this repository. Users download a
  Gemma GGUF themselves under its own terms.
- llama.cpp is not vendored. Users build it themselves, by default at the
  tested release.
- The request and response shape follows TypeSafe's publicly documented Jev
  interface. This project is independent and is not affiliated with or
  endorsed by TypeSafe.
