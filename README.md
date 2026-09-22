# riderless

A local decision API that answers structured questions by reading logits, with
zero generated tokens. You supply a `state` and a named map of questions; the
service returns one typed answer per question: a Choice, a Score, or a Noul
(true/false probability).

It is not a chat model, not a text generator, and not a drop-in replacement for
a hosted service. It runs one llama.cpp child process on your own machine,
answers one request at a time, and never samples or appends a token.

## Why riderless

Jonathan Haidt describes the mind as a rider on an elephant: the elephant is the
fast, automatic part that actually moves, and the rider is the conscious
narrator who explains afterwards where they were going (The Happiness
Hypothesis, 2006, reused in The Righteous Mind). The elephant is roughly what
Kahneman calls System 1. This API reads the decision straight off the model's
final-position logits and never lets it narrate: no explanation, no
after-the-fact story, zero generated tokens. The name is a label for that design
choice, not a claim about how the model works inside.

## The three question types

A request names the model, one state (a string or any JSON value), and up to 32
questions. Question ids are yours; they are correlation metadata and never reach
the prompt.

```json
{
  "model": "local-gemma-riderless-v1",
  "state": {"message": "The card was charged twice"},
  "questions": {
    "route": {
      "type": "choice",
      "instructions": "Choose the best route",
      "criteria": {"billing": null, "technical": "A software defect"}
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

The response carries one answer per question id. Choice returns the argmax
option and the full distribution. Score returns the zero-based expected value
`sum(i * p_i)`, so label probabilities `[0.2, 0.3, 0.5]` give `1.3`. Noul returns
only `P(true)`.

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

Limits: Choice takes 2 to 255 options at the wire boundary and is then rejected
above the backend's validated label capacity (26); Score takes 2 to 10 levels;
Noul needs instructions or a true/false rubric. `output_tokens` is always 0.

## How it works

Each question is compiled into its own prompt from the state plus that
question's own instructions and criteria, with its options labelled `A`, `B`,
`C` and so on. The worker prefills that prompt and reads the final-position
logits at the validated single-token label ids, then softmaxes over those labels
only. There is no sampler, no answer token, no autoregressive loop. The label
alphabet is A to Z, which caps a question at 26 options, and the default context
is 2048 tokens (`--context`).

A question never sees its siblings, so anything a question depends on must be in
the state or in that question's own instructions. Within one request the text
every question shares (the system instruction and the state) is prefilled once:
the first question prefills its whole prompt, and each later question trims the
context back to the shared prefix and prefills only its own remainder. The split
point is computed from the question's own prompt alone and the prefix is reused
only on exact token equality, so a question is computed identically alone,
reordered, or beside any siblings. Nothing is cached between requests.

Questions are still evaluated one at a time by default; the request is not a
single combined forward pass. See
[docs/decisions/0002](docs/decisions/0002-share-state-prefix-within-a-request.md)
for why, and for the cost of the alternative. An opt-in worker mode
(`--batched`) does evaluate a request's questions in one batched decode, which
is faster and gives up that sibling independence:
[docs/decisions/0004](docs/decisions/0004-optional-batched-question-evaluation.md)
and [docs/results/batched-mode.md](docs/results/batched-mode.md).

## Requirements

- An NVIDIA GPU, or CPU if you are patient. With Gemma 4 26B-A4B UD-Q4_K_XL and
  the default 2048-token context, the worker holds about 18 GiB of device memory
  (roughly 16 GiB of weights plus KV and compute buffers), so budget that much
  free VRAM. In the measured run the whole card peaked at 22.5 GiB, but that
  figure includes a 4.83 GiB pre-load baseline from unrelated processes.
- A llama.cpp build. This project is tested against release `v0.4.1` (commit
  `b29c606e28a01b1bc8c1351026a0fa6e616bf6c4`), which the base-runtime script
  fetches by default. Other revisions work if the common library API still
  matches, and are built and hash-verified identically; results may differ,
  and the build and the service each say so once. Nothing from llama.cpp is
  vendored here; you build it yourself.
- Python 3.12 or newer, Git (the base-runtime script fetches the revision),
  CMake, and a C++ toolchain.
- The Gemma 4 26B-A4B instruction-tuned GGUF. Weights are not included in this
  repository and are not redistributed here; download them yourself under their
  own licence terms.

## Quick start

```bash
# 0. Install the package and its dependencies. Without uv, use
#    `python -m venv .venv && .venv/bin/pip install -e .` and drop the `uv run`
#    prefix below, activating that virtualenv instead.
uv sync

# 1. Build the llama.cpp base runtime (headers, shared libraries, and a
#    build.json that records the revision and per-file hashes). Defaults to the
#    tested release, v0.4.1; --revision <tag-or-sha> builds another one. Drop
#    --cuda for a CPU-only runtime; with it you need a CUDA toolkit. On a small
#    machine also pass --cuda-architectures with your own GPU's compute
#    capability (120 for an RTX 5090): building every architecture costs time
#    and memory you do not need to spend.
uv run python scripts/riderless/build_base_runtime.py \
  --out build/llama-base --cuda --cuda-architectures 120

# 2. Build the worker against that base runtime. The output directory must not
#    already exist; the build records every input hash in build.json and runs its
#    CPU unit test. No model is loaded and no GPU work happens here.
uv run python -m riderless.api.native.build \
  --base build/llama-base \
  --output build/api-worker

# 3a. Answer a file of requests (one JSON object, a JSON array, or JSONL).
uv run python -m riderless.api.cli run \
  --input requests.jsonl --output results.jsonl \
  --worker build/api-worker/build/riderless-worker \
  --manifest build/api-worker/build.json \
  --model-path models/gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf \
  --gpu --diagnostics

# 3b. Or run the ASGI app on loopback for a manual look.
uv run python -m riderless.api.cli serve \
  --worker build/api-worker/build/riderless-worker \
  --manifest build/api-worker/build.json \
  --model-path models/gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf \
  --gpu --host 127.0.0.1 --port 8090
```

The worker and manifest paths above are also `ApiConfig`'s defaults, so you can
omit `--worker` and `--manifest` if you keep this layout. Download the Gemma 4
26B-A4B instruction-tuned GGUF yourself and point `--model-path` at it; the
default expects it at `models/gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf`.

Routes: `POST /v1/decisions`, `GET /v1/models`, `GET /health`, plus FastAPI's
`/openapi.json`. Add `?diagnostics=true` to a request for a per-question readout
(raw label logits, token mapping, allowed-label coverage, full-vocabulary
argmax, prompt hash and version, prompt/processed/reused token counts, timing,
model and runtime hashes, and the `generated_tokens: 0` and
`callbacks_enabled: false` invariants). `GET /v1/models` reports the live limits
and capabilities. A copy of the validated schema is at
[docs/openapi.json](docs/openapi.json). `POST /v1/systemone` is a compatibility
alias that behaves identically to `POST /v1/decisions`, offered as an
interoperability path for clients written against that request shape.

The batch CLI creates its output file only if the path does not exist, and
starts one backend for the whole batch. A JSONL adapter for SemIf-style rows
(`python -m riderless.api.semif to-api|from-api`) preserves row ids and option
order.

## Errors

Errors are `{"error": {"code", "message", "field?", "retryable"}}`. Native
stderr is never copied into an HTTP body.

| Status | Code | Meaning |
| --- | --- | --- |
| 400 | `malformed_json`, `invalid_content_length` | The body is not valid JSON, or the length header is unusable. |
| 404 | `unknown_model` | `model` is not `local-gemma-riderless-v1`. |
| 408 | `timeout` | The request exceeded the configured timeout. Retryable. The child is reaped. |
| 413 | `request_too_large` | Body over the configured limit (1 MiB by default). |
| 415 | `unsupported_media_type` | Content type is not `application/json`. |
| 422 | `invalid_request` | The request does not match the schema. Names the field. |
| 422 | `budget_error` | The compiled request exceeds a model limit, such as the context or option capacity. |
| 422 | `unsupported_content` | Caller text spells a model control token, which could forge a turn boundary. Ordinary markup and angle brackets are fine. |
| 429 | `busy` | One inference request runs at a time. Retryable. |
| 500 | `internal_error` | The worker or the protocol failed. Not retryable. |
| 529 | `unavailable` | No usable backend. Retryable, but there is no automatic restart: the app must be restarted. |

A batch never returns a partial success. Any ambiguous transport failure,
timeout, or cancellation reaps the owned child, and the service then answers 529
until it is restarted.

## Limitations

Read these before trusting an answer.

- **Confidence is not calibrated.** It is `max_probability_v1`, the winning
  label's conditional probability. Across the 1,286 use-case questions of the
  first frozen run, 94.7% of answers exceed 0.99, the mean is 0.995 when right
  and 0.945 when wrong, and only 18 answers fall below 0.8. It must not gate
  decisions without task-specific calibration.
- **Probabilities are conditional on the options you supplied.** High coverage of
  the full-vocabulary mass does not certify a correct answer.
- **Deterministic per configuration, not across configurations.** With a fixed
  build, batch size, and prefix-reuse setting, repeats are bit-identical and a
  question is unaffected by its siblings (72 isolation comparisons differ by
  exactly 0.0). Change the prefill batch shape and about 1% of borderline
  questions move: 4 of 1,286 answers changed when reuse was turned on, and 6 of
  1,286 when only the batch size changed with reuse off.
- **One request at a time.** Concurrency is 1 and a second request gets 429.
  Question count costs time: prefix reuse makes each extra question over a long
  state cost about 35 ms rather than a full reprefill.
- **No listening-server packaging.** The CLI's `serve` is for manual checks. There
  is no supervised service unit, no auth, and no rate limiting here.
- **The quality evidence is synthetic.** The seven use-case suites are original
  hand-authored cases with gold labels frozen before inference, not an
  established public benchmark. One suite is saturated and another sits close to
  ceiling.
- **A question cannot see its siblings.** Shared policy text must live in the
  state or be repeated per question. This trap cost real accuracy in the
  guardrail suite.

## Measured results

All numbers come from runs on one RTX 5090 with Gemma 4 26B-A4B UD-Q4_K_XL, and
every one of them is reproduced with its context in `docs/results/`. Those runs
were recorded before this project was renamed, under the earlier internal model
id `local-gemma-systemone-v1` and the worker binary name that went with it; the
rename changed identifiers only, and the compiled prompt text is byte-identical
across it.

Those runs were also measured on llama.cpp revision `afeebe1`, which was the
pinned revision at the time. The tested revision is now release `v0.4.1`. Each
results page says which revision produced its numbers, and the re-validation on
v0.4.1 is reported separately rather than written over them.

- Conformance and isolation: 36 of 36 hand-authored smoke answers, 0 generated
  tokens on every request, 72 isolation comparisons at exactly 0.0 delta, and a
  negative-control worker with the context clear removed fails the same harness.
  [docs/results/api-validation.md](docs/results/api-validation.md)
- Task quality: seven use-case suites, 1,286 questions, 0.928 accuracy against a
  0.694 trivial baseline after the label revision (0.925 against 0.693 on the
  first frozen run), with the auditors' caveats on what that does and does not
  support.
  [docs/results/usecase-suites.md](docs/results/usecase-suites.md)
- Latency: about 30 ms per 111-token question without prefix reuse; with reuse,
  eight questions over a 1,007-token state take 399 ms against about 1,100 ms
  without it (that no-reuse figure is extrapolated from measured linear
  scaling), and the seven suites' median request goes from 247 ms to 194 ms.
  [docs/results/prefix-reuse.md](docs/results/prefix-reuse.md)
- Build-to-build comparison: adding a runtime invariant check to the worker left
  every answer, probability, and raw logit identical, and its cost is below the
  noise floor of the measurement.
  [docs/results/worker-build-comparison.md](docs/results/worker-build-comparison.md)
- Re-validation on llama.cpp v0.4.1: the same harness and the same suites run
  against the current tested revision, compared question by question with the
  `afeebe1` numbers above.
  [docs/results/llama-v0.4.1-revalidation.md](docs/results/llama-v0.4.1-revalidation.md)
- Opt-in batched evaluation: the same suites and harness with every question of
  a request in one decode, against the sequential default on the same worker.
  [docs/results/batched-mode.md](docs/results/batched-mode.md)
- Against the same model used generatively: llama-bench prefill and decode
  figures for the identical GGUF, and what they imply for one question and for
  eight questions over one document.
  [docs/whitepaper.md](docs/whitepaper.md)

Further reading: [docs/whitepaper.md](docs/whitepaper.md) for the whole
story in one place, [docs/architecture.md](docs/architecture.md) for the process
model and protocol, [docs/usecase-suites.md](docs/usecase-suites.md) for the
suites and how to run them, and [docs/decisions/](docs/decisions/) for the four
decision records that shape v1.

## Licence and attribution

Apache-2.0. See [LICENSE](LICENSE).

The request and response shape follows TypeSafe's publicly documented Jev
interface so that callers can reuse a familiar payload. This project is
independent, and is not affiliated with, endorsed by, or certified by TypeSafe.
No parity with any hosted service is claimed or measured here.

llama.cpp is MIT licensed and is built by you; no
llama.cpp source is vendored in this repository. Model weights are not included:
you download Gemma 4 yourself and use it under its own terms.
