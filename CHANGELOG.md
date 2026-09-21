# Changelog

Notable changes to systemone-local. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
intends to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
once it tags a release.

The project has no tagged release yet, so everything lives under Unreleased.
Until the first tag, the HTTP contract, the CLI flags, and the native worker
protocol can change without a deprecation period.

## Unreleased

### Added

- `systemone` Python package: a local, non-generative decision API. One owned
  llama.cpp child process prefills a compiled prompt and reads final-position
  logits over single-token labels, so a request produces zero generated
  tokens.
- Three caller-defined question types, batched in one request: `choice`
  (pick one of a caller-supplied label set), `score` (a bounded numeric
  rating), and `noul` (true or false against caller criteria).
- HTTP surface: `POST /v1/systemone`, `GET /v1/models`, and `GET /health`,
  served by a FastAPI application whose lifespan owns the worker child.
  Importing the package or building the app starts no process.
- `systemone.api.cli` with `run` (evaluate a JSON or JSONL file of requests
  into JSONL, no port needed) and `serve` (run the ASGI app on a loopback
  listener).
- SemIf row import and export helpers in `systemone.api.semif`.
- Native worker under `systemone/api/native`: a C++ worker built against a
  user-provided llama.cpp checkout, plus `systemone.api.native.build`, which
  verifies the frozen runtime before compiling and records source, executable,
  runtime, and llama.cpp revision hashes in a build manifest. Startup pins the
  worker executable, the runtime files and bundle, and the llama.cpp revision;
  the model file's SHA-256 is always computed and is enforced when
  `--model-sha256` is configured.
- Validation and use-case scripts under `scripts/systemone`, with the
  matching request and expectation fixtures under `systemone/examples`.
- Architecture decision records under `docs/decisions` and measured results
  under `docs/results`.
- Community and governance files: Apache-2.0 license, NOTICE, contribution
  guide with DCO sign-off, Code of Conduct, security policy, issue and pull
  request templates, and a CPU-only CI workflow.

### Notes

- Model weights are not distributed with this repository. Users download a
  Gemma GGUF themselves under its own terms.
- llama.cpp is not vendored. Users build it themselves at a pinned revision.
- The request and response shape follows TypeSafe's publicly documented Jev
  interface. This project is independent and is not affiliated with or
  endorsed by TypeSafe.
