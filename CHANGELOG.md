# Changelog

Notable changes to riderless. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

While the major version is 0, the HTTP contract, the CLI flags, and the native
worker protocol can change between minor versions without a deprecation
period. Each release records which llama.cpp revision and which model file its
published measurements were taken on.

## Unreleased

### Added

- Prebuilt worker bundles, published from GitHub Releases, so a first answer
  needs no compiler. `python -m riderless.api.cli worker fetch` (also
  `scripts/riderless/fetch_worker.py`) picks `cuda13`, `cuda12` or `cpu` from
  the NVIDIA driver `nvidia-smi` reports and says which, verifies the download
  against the release's `SHA256SUMS`, checks GitHub's build provenance with
  `gh attestation verify` when `gh` is installed (`--require-attestation`
  makes a missing or failing check fatal instead of a warning), and unpacks
  into a `--output` that must not already exist
  ([ADR 0005](docs/decisions/0005-relocatable-prebuilt-worker-bundles.md)).
- `riderless.api.native.bundle`, which defines the install layout
  (`build.json`, `build/riderless-worker`, `runtime/*.so*`) and packs and
  unpacks it. `ApiConfig`'s defaults already point inside it, so a bundle
  unpacked into `build/api-worker` needs no flags.
- A release workflow on `v*` tags building all three flavors in NVIDIA's devel
  containers with `GGML_NATIVE=OFF` and
  `CMAKE_CUDA_ARCHITECTURES=80;86;89;90;120`, attaching the tarballs and a
  `SHA256SUMS` to the release, and attesting them with
  `actions/attest-build-provenance`.
- A CUDA 13 container image, `ghcr.io/potto007/riderless:<version>-cuda13`,
  built from `docker/Dockerfile`. It expects a GGUF mounted at `/models` and
  serves on 8090.
- `build_base_runtime.py --no-native` (`GGML_NATIVE=OFF`), and CUDA builds now
  copy `libcudart`, `libcublas` and `libcublasLt` next to the CUDA backend and
  hash them with the rest, so a machine with only the NVIDIA driver can run
  the result. `--no-cuda-redist` skips that.

### Changed

- **Build manifests are schema 2 and relocatable.** `executable` and
  `runtime_dir` are now relative to the manifest's own directory and resolved
  against it; an absolute path, or a relative one escaping the bundle, is
  refused. `base_build`, which recorded the builder's own path, is gone;
  `base_build_json_sha256` keeps its provenance. Schema 1 manifests are still
  read with their absolute paths, so a pre-0.2.0 install keeps starting, but
  it cannot be packed or moved. Every hash check is unchanged.
- **The worker takes `--runtime-dir`** instead of a compile-time backend
  directory, and the backend passes the directory it just hashed. The worker
  links with an `$ORIGIN/../runtime` run path and no absolute path, so its
  sha256 no longer depends on where it was built. `riderless-so1-probe` takes
  the same argument, and `bench_competitor_so1.py` gained `--runtime-dir`.
- `riderless.api.native.build` copies the base runtime into the bundle and
  re-hashes the copy, builds in a scratch directory it removes on success, and
  leaves `build/` holding only the executable.

## 0.1.0 - 2026-09-22

First tagged release. Every measurement in `docs/results` was taken on
llama.cpp release `v0.4.1` (commit `b29c606e28a01b1bc8c1351026a0fa6e616bf6c4`)
with the Unsloth Gemma 4 26B-A4B UD-Q4_K_XL GGUF on one RTX 5090, unless the
page says otherwise.

### Added

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
- Shared-prefix reuse within a request. The compiler reports the byte length
  of the prefix the questions of a request have in common, and the worker
  keeps that prefix's KV cells across the questions instead of re-prefilling
  them, with an exact 0.0 isolation check in the validation harness
  ([ADR 0002](docs/decisions/0002-share-state-prefix-within-a-request.md),
  [results](docs/results/prefix-reuse.md)).
- Opt-in batched question evaluation. `ApiConfig.batched` (CLI, runner and
  worker `--batched`) gives every question of a request its own KV sequence over
  one copy of the shared prefix and decodes all the remainders together.
  `ApiConfig.batched_context` (worker `--batched-context`, default 8192 cells)
  sizes the unified KV cache and may not exceed `max_questions * context_size`.
  The default stays sequential and reproduces the recorded v0.4.1 suite run bit
  for bit
  ([ADR 0004](docs/decisions/0004-optional-batched-question-evaluation.md),
  [results](docs/results/batched-mode.md)).
- A batched worker names the regime that answered each request in the public
  response body as `evaluation: {mode, fallback}`, and a request that needs more
  KV cells than `batched_context` is answered by the sequential path with
  `fallback: "context"` rather than being truncated or refused. Diagnostics gain
  `evaluation_mode` and `batch_sequences` per question. `GET /v1/models` reports
  `runtime.batched` and `runtime.batched_context`.
- A tested llama.cpp default instead of a hard pin. The build records release
  `v0.4.1` as `TESTED_LLAMA_TAG` and `TESTED_LLAMA_REVISION`,
  `build_base_runtime.py` fetches that tag by default and verifies that it
  still resolves to the recorded commit, `--revision <tag-or-sha>` builds any
  other revision, and an untested revision logs one warning at startup instead
  of being refused. `GET /v1/models` reports `runtime.llama_revision` and
  `runtime.tested_revision`
  ([ADR 0003](docs/decisions/0003-tested-default-revision-instead-of-a-hard-pin.md),
  [re-validation results](docs/results/llama-v0.4.1-revalidation.md)).
- `build_base_runtime.py --cuda-architectures`, passed through as
  `CMAKE_CUDA_ARCHITECTURES`. Omitted, llama.cpp's own default applies.
- Validation and use-case scripts under `scripts/riderless`, with the
  matching request and expectation fixtures under `riderless/examples`:
  seven hand-written use-case suites (241 cases, 1,286 questions) with frozen
  gold labels, a validation harness with sibling-independence, repeat-identity
  and cross-question contamination probes, and `compare_observations.py` and
  `compare_fallback.py`, which reduce two recorded runs to answer changes,
  probability and raw logit moves, tokens and latency.
- An open-model comparison. `docs/research/open-model-survey.md` reviews
  eight open decision-readout projects for ideas worth adopting, and
  `docs/results/open-model-comparison.md` runs three of them on the seven
  suites with the same scorer: Kev-9B and Laya as shipped, and so1
  (open-alternative-jev) on riderless's own GGUF and llama.cpp build through a
  backend written for the purpose (`riderless/api/native/so1_probe.cpp`), in
  both its separate and packed modes. The drivers
  (`bench_competitor_kev.py`, `bench_competitor_laya.py`,
  `bench_competitor_so1.py`, `compare_competitor_outcomes.py`) and a
  page-cache eviction helper (`evict_file_cache.py`) ship under
  `scripts/riderless`.
- A whitepaper (`docs/whitepaper.md`) covering the design, the measured
  results, and a generative baseline measured with llama-bench on the
  identical GGUF.
- Architecture decision records under `docs/decisions` and measured results
  under `docs/results`.
- Community and governance files: Apache-2.0 license, NOTICE, contribution
  guide with DCO sign-off, Code of Conduct, security policy, issue and pull
  request templates, and a CPU-only CI workflow on Python 3.12 and 3.13 with a
  native helper build test.

### Notes

- Model weights are not distributed with this repository. Users download a
  Gemma GGUF themselves under its own terms. The worker accepts the `gemma4`
  architecture only.
- llama.cpp is not vendored. Users build it themselves, by default at the
  tested release.
- Reported confidences are the raw softmax over label logits and are not
  calibrated: 94.7% of the 1,286 suite answers sit above 0.99, including most
  of the wrong ones. Do not gate a decision on the probability without fitting
  a temperature first.
- Answers are deterministic per configuration only. About 1% of borderline
  answers move with batch size, prefix reuse, or llama.cpp revision.
- The request and response shape follows TypeSafe's publicly documented Jev
  interface. This project is independent and is not affiliated with or
  endorsed by TypeSafe, and no measurement of the hosted service appears in
  this repository.
