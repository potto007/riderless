# API validation: conformance, isolation, failure paths, latency

Run on 2026-09-21 on one RTX 5090 with Gemma 4 26B-A4B UD-Q4_K_XL. The harness
drives the real ASGI app over an in-process transport with one owned native
child. No port is bound.

Every number here is read from a recorded run. Raw evidence directories are not
part of this repository.

## What was validated

The validation protocol and the 36-question fixture
([`unridden/examples/api-v1-cases.json`](../../unridden/examples/api-v1-cases.json),
12 requests: 17 Choice, 14 Noul, 5 Score) were frozen before any inference for
this API, and the fixture hash is recorded before the model loads. The harness
is [`scripts/unridden/validate_api_v1.py`](../../scripts/unridden/validate_api_v1.py);
failure paths are driven by
[`scripts/unridden/validate_api_failures.py`](../../scripts/unridden/validate_api_failures.py).

Runtime configuration for every run below: context 2048, batch 256, ubatch 256,
8 threads, all 31 layers offloaded to the GPU, causal attention, CUDA fusion and
graphs on, callbacks off. The model file is hashed at startup and refused if it
differs from the configured pin.

**Every number on this page was measured on llama.cpp revision
`afeebe103bd99cda8f5dfaefcabadf890db7fda7`**, which was this project's pinned
revision at the time. The tested revision is now release `v0.4.1`. The same
harness was re-run on v0.4.1 and the two runs are compared in
[llama-v0.4.1-revalidation.md](llama-v0.4.1-revalidation.md); the tables below
are left as measured rather than reattributed.

Limits reported by the live `GET /v1/models`: 26 backend Choice options
(validated single-token labels A to Z; the wire protocol accepts up to 255 and
anything larger is rejected with 422), 10 Score levels, 2048 context tokens, 32
questions, 1 MiB request body, concurrency 1.

## Results of the authoritative run

| Check | Result |
| --- | --- |
| Requests | 122 recorded: 112 successful, 10 expected rejections |
| Semantic smoke | 36/36 (Choice 17/17, Score 5/5, Noul 14/14) |
| Generated tokens | 0 across all 112 successes; `output_tokens` is always 0 |
| Executed input tokens | 49,670 |
| Isolation | 72 comparisons (solo, reversed, id-renamed, repeated); max probability delta 0.0, max raw-logit delta 0.0 |
| Allowed-label coverage | minimum 0.99997 of full-vocabulary mass; the unrestricted argmax was an allowed label in 246 of 246 questions |
| Rejections | 404 unknown model; 422 unknown field, empty questions, question limit; 422 `budget_error` context limit; 422 `unsupported_content` forged control token in the state; 400 malformed JSON; 415 media; 413 body; 429 concurrent overlap |
| Accepted on purpose | A state containing HTML, `[INST]`, and an older model generation's literal end-of-turn marker: plain text here, answered correctly |
| BOS | The worker's doubled-BOS guard never fired |
| Recovery | health `ok` after the rejections; an overlapped request recovered |
| Shutdown | `/health` returned 529 `unavailable`; the exact owned child was gone |
| Startup | 9.6 s from app start to ready with a warm page cache (16.1 s on the first cold run), including model hashing and load |

36 of 36 on small hand-authored fixtures with explicit facts is a smoke result.
It establishes neither broad accuracy nor calibration.

A later run of the same harness on the worker build with shared-prefix reuse
recorded 134 requests, the same 36 of 36 smoke answers, 0 generated tokens, and
the same 72 isolation comparisons at exactly 0.0. See
[prefix-reuse.md](prefix-reuse.md).

## Negative control

A worker was built from a copy of the final source with the per-question
`llama_memory_clear` removed. The unchanged harness failed at the first mixed
versus solo comparison, after 12 requests, with a probability delta of 5.6e-4
against the 1e-6 tolerance. So the 72 comparisons at exactly 0.0 are a real
isolation result: the harness does catch carried-over context.

That run also showed that the worker's `cache_cleared: true` diagnostic had been
a hardcoded literal, since the no-clear worker still reported it. The worker now
checks that sequence 0 holds no positions before a full prefill and fails the
request otherwise, so the field is a measured fact.

## Timeout and worker-death paths on the real GPU

Two app lifespans, one owned child each:

| Scenario | Result |
| --- | --- |
| Request timeout 50 ms against an 8-question, roughly 1,230-token-per-question request | 408 `timeout`, retryable. Owned child gone. `/health` 529. Next request 529, retryable. |
| The exact owned child killed with SIGKILL between requests | 500 `internal_error`, not retryable, no path or stderr in the body. `/health` 529. Next request 529. |

This is the "remain explicitly unavailable until restart" branch, observed with
the real worker rather than only a fake one.

## CLI and SemIf adapter

Run through `python -m unridden.api.cli run --gpu --diagnostics`, sequentially,
one model load per invocation. Every answer was bit-identical to an earlier run
on a previous worker build.

| Input | Responses | Answers | Input tokens | Output tokens |
| --- | --- | --- | --- | --- |
| 2-request JSONL batch (one load) | 2 | 4 Choice, 1 Noul, 1 Score | 936 | 0 |
| Single mixed JSON request | 1 | 1 Choice, 1 Noul, 1 Score | 438 | 0 |
| Converted SemIf rows | 2 | 2 Choice | 194 | 0 |

Every question's diagnostics reported `generated_tokens: 0` and
`cache_cleared: true`. Rerunning onto an existing output file was refused before
any model load. The SemIf `to-api` then `from-api` roundtrip reproduced both rows
exactly, including ids and option order. A deliberately wrong `--model-sha256`
was refused at startup before any model load.

## CPU checks at the validated commit

The checks this repository runs, which are also the ones in
`.github/workflows/ci.yml`, pass: `ruff format --check .` and `ruff check .`
(line length 88, rules `E,F,I,UP,B,SIM`), `mypy --strict
--explicit-package-bases unridden scripts`, and `pytest`, which is 55 tests
across `unridden/tests` (53 at the validated commit, plus two that cover the
`POST /v1/systemone` compatibility alias added with the rename). The larger
validation run above was recorded against the originating tree, which carried
additional legacy modules and their tests; only the API, its tests, and the
harness were ported here.

## Independent review

A separate agent that did not write the code reviewed it read-only against the
first passing evidence run. Verdict: approve with fixes. It reproduced the
request accounting independently, audited `worker.cpp` and confirmed that the
only tokens any `llama_decode` receives are prompt tokens, and judged the exact
0.0 isolation delta plausible rather than a comparison artefact.

| Finding | Severity | Resolution |
| --- | --- | --- |
| Caller text parsed with `parse_special=true` could forge turn tokens ahead of the readout | important | The worker rejects control and end-of-generation tokens in caller content: 422 `unsupported_content`. Proven on GPU with the model's real markers. |
| The model hash was self-computed, never compared to an expected value | important | `ApiConfig.model_sha256` pin, CLI `--model-sha256`, startup refuses a mismatch. |
| An oversized stderr line killed the drain task and turned a timeout into a 422 | important | The drain survives oversize lines; teardown never lets a drain error replace the real one. |
| An oversized stdout line raised a bare `ValueError` and left the child unreaped | important | Mapped to a protocol error that reaps the child. |
| Media type compared case-sensitively | minor | Lowercased. |
| Every worker preflight failure became a client 422 | minor | The worker names the reason; only budget and control tokens are 422, a compiler and worker disagreement is 500. |
| The request cap could exceed the worker's fixed 4 MiB line limit | minor | The Python side clamps the outgoing envelope to the worker constant. |
| The reap-on-timeout contract was unstated | minor | Stated on the `Backend` protocol. |
| No catch-all in `NativeBackend.evaluate` | minor | Unknown failures now reap the child. |

Two gaps the reviewer left open were closed afterwards: the negative control and
the real-GPU 408, 500, and mid-run 529 paths, both above. Still open: the file
and manifest provenance checks lack negative unit tests; per-chunk prefill sizes
are summed rather than reported individually; `/v1/models` returns 200 with an
empty list before the backend is ready. The fixes above were not re-reviewed
independently.

## Latency and resource use

One GPU, otherwise idle, without shared-prefix reuse. Three repeats per point,
so p95 is descriptive only. API time is the in-process ASGI round trip; native
time is the worker's own per-request total.

| Axis | Size | Input tokens | API median ms | Native median ms |
| --- | --- | --- | --- | --- |
| questions | 1 | 111 | 32.0 | 29.0 |
| questions | 2 | 222 | 64.7 | 60.0 |
| questions | 4 | 444 | 126.2 | 119.1 |
| questions | 8 | 888 | 254.6 | 240.9 |
| padding words | 0 | 111 | 32.1 | 29.0 |
| padding words | 384 | 495 | 74.8 | 68.1 |
| padding words | 896 | 1007 | 133.9 | 122.9 |
| padding words | 1728 | 1839 | 244.0 | 223.8 |

Without reuse, cost is linear in question count (about 30 ms per 111-token
question) because each question gets a fresh full prefill. Input length scales
sublinearly per token: 1,839 tokens cost about 7.6x the 111-token case. Across
all 112 successful calls the API median was 97 ms and p95 247 ms. Python and
transport overhead is roughly 3 to 15 ms per call. Shared-prefix reuse changes
the question-count axis substantially; see [prefix-reuse.md](prefix-reuse.md).

GPU memory went from 4.83 GiB before load to a 22.5 GiB peak (about 17.6 GiB for
the worker: 16.2 GiB of weights plus KV and compute buffers) on a 32.6 GiB card,
and returned to 4.85 GiB after shutdown.

## Coverage of the documented query patterns

The Jev interface documents a set of composition patterns. This table records
which local fixture exercises each one and where the boundary of this delivery
sits. It is a scope statement about this project, not a comparison with any
hosted service.

| Documented composition | Local fixture or check | Boundary of this delivery |
| --- | --- | --- |
| Repeated Noul decisions | Noul cases repeated after unrelated requests | Probability stability; no broad uncertainty calibration. |
| Repeated Choice decisions | Choice cases repeated after unrelated requests | Choice and raw-logit stability; no moderation benchmark. |
| Mixed questions over one state | Every mixed fixture compared with standalone and reordered questions | Independent results, serial execution. No parallel latency is claimed. |
| Reranking | `passage-ranking`, `diagnostic-next-check` | Supplied candidate distributions and binary relevance; retrieval and shortlist construction remain caller code. |
| Semantic line search | `passage-ranking` | Candidate selection plus an answer-present gate; large line inventories are constrained by the option capacity. |
| Structure recovery | Structured states and rubrics, speculative companion questions in `function-and-arguments` | The primitives compose; adjacent-line merging and document reconstruction are not implemented. |
| Function calling | `function-and-arguments` | Selects a function name and closed arguments. No function is executed. |
| Skill suggestion | `diagnostic-next-check` | Small candidate choice and evidence checks; no full catalogue or staged selection service. |
| Entity alignment | `entity-alignment` | Independent match criteria and an ordered level distribution. |
| RAG passage classification | `passage-ranking`, `citation-relations` | Supplied passage relevance and evidence checks; retrieval and answer generation stay outside. |
| Citation checking | `citation-relations` | Support, contradict, and silent relations over supplied facts; no citation crawler or quote lookup. |
| Guardrails | `message-policy-gates` | Finite policy judgments; production thresholds and safety efficacy are not established. |
| Structured extraction cascade | `candidate-span-extraction`, `structured-expense-policy` | Candidate and field judgments; no external model escalation. |
| Date extraction | `date-components` | Supplied components and mode selection; calendar arithmetic remains deterministic caller code. |
| Candidate-span extraction | `candidate-span-extraction` | Selects supplied spans including an absent option; cannot invent a new string. |
| Hierarchical classification | `taxonomy-frontier` | Node and frontier judgments; tree traversal and beam search remain caller code. |
| Feature discovery | `review-features` | Answers supplied feature questions; does not propose new questions or train a downstream model. |
| Confidence-gated broad or fine classification | `taxonomy-frontier`, `ticket-temporal-status` | Full distributions are available to caller policy; max probability is not calibrated correctness. |

## Limits of this evidence

Finite-label probabilities are conditional on the supplied options. Coverage
diagnostics show how much full-vocabulary mass those labels receive. Neither
high coverage nor a large winning probability certifies factual correctness.
Task quality beyond this smoke set is reported in
[usecase-suites.md](usecase-suites.md).
