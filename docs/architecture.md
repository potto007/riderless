# Architecture

How a request becomes a set of label probabilities, and what keeps that process
auditable. Decisions behind it: [0001](decisions/0001-full-depth-label-readout-in-an-owned-child.md),
[0002](decisions/0002-share-state-prefix-within-a-request.md) and
[0003](decisions/0003-tested-default-revision-instead-of-a-hard-pin.md).

## Process model

```
HTTP client  ->  FastAPI app (riderless/api/app.py)
                   ApiService: single-flight guard, timeouts
                     compiler.py   -> one prompt per question
                     native_backend.py -> JSONL over stdio
                       native/worker.cpp (one owned child, llama.cpp linked)
                     mapping.py    -> typed answers and diagnostics
```

- **Importing the package starts nothing.** `create_app()` builds the app
  without loading a model or touching the GPU. The ASGI lifespan owns the
  child: it starts on startup and is closed on shutdown. The batch CLI does the
  same thing around one batch.
- **Exactly one child process.** `NativeBackend` spawns the worker executable
  with the model path, both provenance hashes, and the runtime sizes as
  arguments, and sets `LD_LIBRARY_PATH` to the pinned runtime directory it
  verified. `GPU` is an explicit opt-in flag; without it the worker runs on CPU.
- **One request at a time.** `ApiService` holds a busy flag; a second concurrent
  request gets 429 without reaching the backend. The worker context is created
  with a single sequence.
- **No automatic restart.** Any ambiguous transport failure, timeout,
  cancellation, protocol violation, or unknown exception reaps that exact child
  and marks the backend unavailable. `/health` and every later request answer
  529 until the app is restarted. This is deliberate: a silent restart would
  reload about 17 GiB onto a GPU with nobody checking headroom.
- **Worker stderr never reaches a client.** It is drained into a bounded tail of
  the last 200 lines and logged at DEBUG under the `riderless.api.native`
  logger. HTTP error bodies carry a code, a message, an optional field name, and
  a retryable flag, and nothing else.
- **A hostile environment is refused at startup.** Both the Python side and the
  worker reject environment variables that would change the computation, such as
  an early-exit layer override or the CUDA fusion and graph disables.

## Prompt compilation

`compiler.py` turns each question into a standalone prompt:

```
<system instruction>

STATE:
<state, rendered as text or as canonical JSON>

QUESTION:
<one heading per question type>

INSTRUCTIONS:
<the question's own instructions, if any>

OPTIONS:
A: <option id> - <description>
B: ...

Reply with one option label only.
```

followed by the fixed answer prefix `Answer:\n`. Option ids and order are
preserved exactly as supplied. Score levels become option ids `0..n-1` with a
legend returned alongside the answer. A Noul becomes a two-option question over
`true` and `false`. Question ids never appear in a prompt.

Non-string states are rendered with sorted keys and compact separators, so the
same state object always produces the same prompt. The prompt text is hashed
into `prompt_sha256`, and a `prompt_version` string is checked by the worker,
which refuses a prompt shape it was not built for.

## The readout

The worker tokenizes the prompt with special-token parsing (the chat template
needs it), checks for exactly one BOS, and rejects any caller text that spells a
control or end-of-generation token. It prefills the prompt in fixed chunks and
reads the final-position logits, then takes the logits at the label token ids
and softmaxes over those only.

The label alphabet is validated against the loaded tokenizer at startup: each
letter from A upward must append exactly one token to a probe prompt, and the
run stops if fewer than 10 such labels exist. That validated list is the
backend's option capacity (26 with the validated model) and it is checked again
per prompt, because a prompt-specific tokenization change would silently move
the readout.

Diagnostics report the allowed-label mass as a fraction of the full-vocabulary
softmax, and the unrestricted argmax token, so a caller can see how much of the
model's probability mass the option set actually captured.

## Isolation rules

These are the invariants the validation harness exercises. The result for a
question must not depend on which other questions travelled with it.

1. A question's prompt contains only the state and that question's own
   instructions and criteria. Siblings and question ids are invisible to it.
2. Before a question that does not reuse a prefix, the context memory is cleared
   and then checked to hold no positions. Before a reusing question, it must hold
   exactly the prefix and nothing else. Both checks are hard failures, which is
   why `cache_cleared` and `reused_tokens` are measured facts rather than claims.
3. The shared-prefix split point is computed from the question's own prompt
   only, and the prefix is reused only when the cached tokens equal it exactly.
   The prefix is therefore always prefilled in the same chunks.
4. A shared prefix under 128 tokens is not split at all.
5. The cached prefix lives in a local variable for the duration of one request.
   Nothing survives into the next request, so no caller's state can leak into
   another's.
6. `processed_tokens + reused_tokens == prompt_tokens` is validated on both
   sides of the protocol.

Measured consequence: 72 comparisons of the same question asked solo, mixed,
reversed, renamed, and repeated differ by exactly 0.0 in probability and in raw
logits, while a negative-control worker with the context clear removed fails the
same harness. See [results/api-validation.md](results/api-validation.md).

Determinism holds per configuration. Changing the prefill batch size or turning
reuse on or off changes batch shapes, and about 1% of borderline questions move;
see [results/prefix-reuse.md](results/prefix-reuse.md).

## JSONL protocol, version 2

One JSON object per line in each direction over the child's stdin and stdout.
Every line is bounded at 4 MiB on both sides; a longer line is a protocol error
that reaps the child. The Python side clamps its outgoing envelope to the same
constant that the worker enforces.

**Handshake.** The worker's first line is a `hello`:

```json
{"type": "hello", "protocol_version": 2, "model_id": "local-gemma-riderless-v1",
 "model_name": "...", "model_sha256": "...", "runtime_sha256": "...",
 "labels": ["A", "B", "..."], "label_token_ids": [1, 2],
 "context_size": 2048, "batch_size": 256, "ubatch_size": 256, "threads": 8,
 "max_questions": 32, "generated_tokens": 0, "callbacks_enabled": false,
 "execution_mode": "full"}
```

The backend refuses to proceed unless the protocol version is 2 and every field
matches what it asked for, including both hashes.

**Request.** One envelope per API request, carrying every compiled question:

```json
{"type": "evaluate", "id": "<correlation id>",
 "questions": [{"id": "route", "messages": [{"role": "user", "content": "..."}],
                "answer_prefix": "Answer:\n", "labels": ["A", "B"],
                "prompt_version": "riderless-gemma-choice-v1",
                "shared_prefix_bytes": 512}]}
```

`shared_prefix_bytes` is a hint about the leading text that every question of
this request shares. Any value below the prompt length is correct; a poor one
just reuses less. Setting it to 0 disables reuse for that question.

**Result.**

```json
{"type": "result", "id": "<correlation id>", "model_sha256": "...",
 "runtime_sha256": "...", "generated_tokens": 0, "callbacks_enabled": false,
 "execution_mode": "full",
 "questions": [{"id": "route", "label_logits": [...], "label_token_ids": [...],
                "allowed_label_mass": 0.99998,
                "full_vocabulary_argmax": {"token_id": 235280, "logit": 21.5},
                "prompt_sha256": "...", "prompt_tokens": 384,
                "processed_tokens": 384, "reused_tokens": 0,
                "cache_cleared": true, "timing_ms": 29.4}]}
```

**Preflight error.** A request the worker will not run at all:

```json
{"type": "error", "id": "<correlation id>", "code": "invalid_request",
 "reason": "budget" | "control_tokens" | "internal"}
```

`budget` and `control_tokens` are the caller's fault and become 422s.
`internal` means the compiler and the worker disagree, which is a 500; the child
stays up, because it answered in protocol.

**Correlation.** Every envelope carries a fresh correlation id and the reply must
echo it. A mismatch means the stream is desynchronized, so the child is reaped.
The result is revalidated against the request (same question ids, same counts,
finite logits, token accounting) before any answer is returned; a mismatch also
reaps the child.

## Provenance manifests

The worker is built by `python -m riderless.api.native.build`, which is
create-only: it refuses an output directory that already exists, so a build can
never be silently overwritten. Before compiling it validates the base runtime:
the base manifest must name a llama.cpp revision, the headers and shared
libraries must be present, and every library must hash to the value recorded in
the base manifest. It then compiles, and runs the CPU unit test for the worker
helpers, which loads no model and does no GPU work.

The resulting `build.json` records:

- the llama.cpp revision the base was built from, the ref that was asked for,
  whether that is the tested revision, and the base build it linked against,
- every runtime library hash plus a single bundle hash over all of them,
- the source hashes of the worker sources and of the SHA-256 helper that the
  build compiles out of your llama.cpp checkout (no llama.cpp file is vendored
  in this repository),
- the executable path and hash,
- the default runtime configuration (context, batch, ubatch, threads, attention
  type, fusion and graph flags) as a build record: these are the build's
  defaults, not a constraint on how the worker is later started,
- the three invariants `generated_tokens: 0`, `callbacks_enabled: false`,
  `execution_mode: "full"`.

At startup the backend rechecks the executable path and hash, every runtime file
hash, the bundle hash, and those three invariants, and refuses to start on any
mismatch. The invariants are checked against the handshake the
worker reports for the flags the backend actually passed, not against the
manifest's recorded `runtime_config`: starting with `--context 4096` leaves the
manifest saying 2048 and raises no mismatch. The source and helper hashes are a
build record and are not rechecked at startup.

The model file is hashed at startup either way. It is pinned only if you
configure a SHA-256 (`ApiConfig.model_sha256`, CLI `--model-sha256`); with a pin
set, any other file is refused before the model is loaded, and unset (the
default) the backend accepts whatever `model_path` points at.

Both hashes travel through the handshake into every response's diagnostics, so
an archived answer says exactly which executable, which runtime bundle, and
which model file produced it.

### Which llama.cpp, and why it is not a gate

The recorded revision is provenance, not integrity. Every hash above is checked
whatever the revision turns out to be, so a build from another llama.cpp is
verified exactly as strictly as the tested one. What the revision buys is the
ability to say which llama.cpp produced a published number.

The project is tested against one release, recorded in
`riderless/api/native/build.py` as both a tag and the commit that tag resolved
to. `build_base_runtime.py` fetches that tag by default; `--revision` builds any
other tag or commit. Asking for the tested tag is checked against the recorded
commit, so a tag that upstream moves fails the build instead of silently
substituting a different tree. Building anything else logs one warning: the
measurements in `docs/results/` were taken on the tested revision and may not
reproduce. It is never a refusal. `GET /v1/models` reports the live values as
`runtime.llama_revision` and `runtime.tested_revision`.

Answers really can move between revisions, because upstream kernels change. See
[results/llama-v0.4.1-revalidation.md](results/llama-v0.4.1-revalidation.md) for
the measured size of that effect, and
[0003](decisions/0003-tested-default-revision-instead-of-a-hard-pin.md) for why
a hard pin was the wrong instrument.
