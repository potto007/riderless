# Unridden: reading decisions off a local LLM without letting it talk

A zero-generated-token decision API built on Gemma 4 26B-A4B and llama.cpp,
what it is for, and how it measures against the same model used generatively.

Paul Otto, 21 September 2026. Repo: github.com/potto007/unridden
(Apache-2.0).

## Abstract

Unridden answers finite questions about a piece of text (which option, how
severe on a scale, true or false) by running one forward pass of a local model
and reading the probability distribution over a handful of single-token labels
at the last position. It never samples a token. On an RTX 5090 a single
111-token question takes 33 ms end to end through HTTP, about the same as the
model's prefill alone, and the answer arrives as a calibrated-looking
distribution rather than text that has to be parsed. On seven synthetic suites
modeled on the use-case families TypeSafe documents for its Jev API, it
scores 0.93 question-level accuracy against a 0.69 trivial baseline. Its
confidence is uncalibrated and saturated near 1.0, and its answers are
bit-identical across runs. The mechanism is cheap, auditable, and has one sharp
limitation: about 1% of borderline answers move when the GPU batch shape
changes, so determinism holds per configuration, not across them.

## What it is

The name comes from Jonathan Haidt's rider and elephant. The elephant is the
fast, automatic mind that actually decides; the rider is the conscious narrator
who explains the decision afterwards and is often wrong about why. Kahneman's
System 1 and System 2 map onto the same split. A generative language model,
asked to classify something, is both at once: it decides somewhere in the
forward pass and then spends most of its compute writing the rider's story.
Unridden keeps the elephant and fires the rider. No token is ever generated.
The API reads the decision directly from the logits and returns it as numbers.

Concretely, it is a FastAPI service that owns one persistent llama.cpp child
process. A request carries a *state* (any text or JSON) and up to 32
*questions* over that state. Each question is one of three kinds:

- Choice: pick one of up to 26 named options. Returns the chosen option, the
  full probability distribution, and a confidence.
- Score: place the state on an ordered scale. Returns the expectation over the
  levels as a real number, plus the distribution.
- Noul: is a stated condition true. Returns P(true).

The request and response shapes follow TypeSafe's publicly documented Jev
interface, so a client written against that shape can point at a local box.
That is the whole relationship: Unridden is independent of TypeSafe and is not
an implementation of their model.

## What it aims to do

The project started from an irritation. Most of what I ask a local model to do
is not writing. It is routing, gating, ranking, extracting, checking. For those
jobs the generated text is overhead: it costs decode steps, it has to be
parsed, it can be malformed, and it hides the model's actual uncertainty behind
a confident sentence. I wanted a decision primitive that was:

1. Cheap. One prefill per question, no decode loop.
2. Typed. The answer is an option id, a number on a scale, or a probability,
   never free text.
3. Auditable. Every response can carry the raw label logits, the prompt hash,
   the token mapping, the fraction of vocabulary mass the allowed labels
   captured, and the hashes of the model and runtime that produced it.
4. Isolated. A question's answer must not depend on which other questions were
   sent with it, in what order, or under what ids.
5. Local and boring to operate. One child process, bounded protocol,
   structured errors, no listening server left behind.

Non-goals, stated so they do not creep back in: it is not a chat endpoint, it
does not do open-ended extraction, it does not claim calibrated probabilities,
and it does not try to beat a hosted model on accuracy. It exists so that a
large share of decision traffic never needs a hosted model at all.

## How it works

### The readout

The compiler renders each question into one user turn: a fixed instruction,
the state, the question heading, the caller's instructions, and the options as
lettered lines (`A: support`, `B: sales - {"when":"new business"}`). It appends
the assistant turn opener and the literal `Answer:\n`. The worker tokenizes that
prompt, prefills it, and reads the logits at the final position. At start-up
the worker verifies once that each candidate letter A through Z tokenizes as
exactly one token after this prompt shape, and it re-verifies per question. The
label logits go through a numerically stable softmax; the argmax is the choice,
the expectation is the score, P(A) is the Noul truth value.

Two guards keep this honest. The worker rejects caller text containing any
control or end-of-generation token, so nobody can forge a turn boundary ahead
of the readout position (Gemma 4's markers are `<|turn>` and `<turn|>`; the
first version of that check looked for the Gemma 2 marker and was caught by a
real run). And every response reports the fraction of the full-vocabulary
probability mass that landed on the allowed labels. In the validation runs the
minimum was 0.99997 and the unrestricted argmax was an allowed label in 246 of
246 questions, which is the evidence that the model is actually answering the
question rather than being forced into a letter.

### Isolation

Each question is compiled into its own prompt. A question never sees its
siblings, so a policy defined in one question is invisible to the next; that
has to live in the state. Inside the worker, the first version cleared the KV
cache before every question. The current version prefills the shared prefix
(instruction plus state) once per request and trims the cache back to it
before each remainder. The split point is computed from the question's own
prompt alone, and reuse happens only on exact token equality, so the chunking
is identical whether a question arrives alone or with 31 siblings. The
validation harness sends every fixture case four extra ways (each question
solo, questions reversed, ids renamed, the whole request repeated) and asserts
the probabilities and raw logits differ by exactly 0.0. They do, on all 72
comparisons, in every run since the first. A negative-control build with the
cache clear removed fails the same harness, which is what makes the passing
result mean something.

### Provenance and errors

The native worker is built create-only into a snapshot directory with a
manifest recording the source hashes, the executable hash, the llama.cpp
revision, and per-library hashes of the runtime. The service hashes the model
file at start-up and refuses to run if it does not match the pinned value. The
handshake carries all of it back, and every diagnostics block repeats the model
and runtime hashes, so a stored answer can be tied to the exact bits that
produced it.

Errors are structured and were exercised on a real GPU, not just against a fake
worker: 400 malformed, 404 unknown model, 408 timeout (the child is killed and
the next request gets 529 until restart), 413 body limit, 415 media type, 422
with three reasons (invalid request, context budget, unsupported content), 429
when a second request overlaps the single inference slot, 500 when the owned
child dies mid-request, 529 when there is no backend. The service admits one
inference at a time by design; on a shared GPU that is a feature.

```
HTTP request  ->  Compiler  ->  Native worker  ->  Gemma 4 26B-A4B, one prefill
(state +          (one prompt     (JSONL over        (... options ... Answer:\n
 questions)        per question)   stdio)             logits at last token,
                                                      labels A..Z only,
                                                      0 tokens generated)
      ^                                                        |
      +---- softmax over label logits, typed answer, ----------+
            distribution, diagnostics, provenance hashes
```

Figure 1. The whole path. The only tokens the model ever sees are the prompt
tokens.

## Task quality

The smoke set (36 hand-written questions) saturated immediately at 36 of 36 and
stopped being informative. To get a real read, seven suites were written to
cover the use-case families TypeSafe documents for Jev: 241 cases, 1,286
questions, with easy, medium and hard tiers. Nobody who wrote a suite ran the
model on it; an independent pass answered everything blind, fixed eight clear
label errors, flagged six contested cases, and the labels were frozen (hashed
into the run manifest) before the first inference. Two adversarial auditors
rescored the output from the raw observations with their own scripts.

| Suite | Questions | Accuracy | Trivial baseline | Auditor's read |
| --- | ---: | ---: | ---: | --- |
| hierarchical, broad-to-fine classification | 151 | 0.921 | 0.417 | Strongest signal. Leaf choice 1.000 against 0.043. |
| candidate span and date extraction | 194 | 0.923 | 0.753 | Good. Best traps in the run. |
| passage rerank and line search | 149 | 0.966 | 0.738 | Good on Choice and Noul; discount the ranking numbers. |
| claim support verification | 96 | 0.958 | 0.677 | Usable but small. |
| tool and argument selection | 122 | 0.984 | 0.738 | Thin but real. Near ceiling. |
| guardrail policy gates | 417 | 0.859 | 0.703 | Too noisy to quote as one score. |
| entity alignment and record matching | 157 | 1.000 | 0.790 | Saturated. Treat as a smoke test. |
| **All** | **1286** | **0.925** | **0.693** | Lift over trivial about +0.23. |

Table 1. First frozen run (worker native-v4). The trivial baseline always
answers the most common gold label for that question id. Two cases were later
revised; the revised run scores 0.928 with baseline 0.694.

The guardrail suite is where the honest problems live. Most misses there are
concentrated in the severity and framing rubrics, which points at the rubrics
as much as at the model. The entity suite says
nothing because everything passes. What I take from the table: the mechanism
works well on classification, extraction-as-choice, and verification, and the
remaining errors are in places where reasonable people disagree about the
label.

## Against the same model used generatively

This is the comparison the project was built to win, so it is worth being
careful about what was measured. Unridden timings are full HTTP round trips
through the ASGI app, measured by the validation harness on native-v8. The
generative figures are from `llama-bench` on the identical GGUF and the same
llama.cpp revision (afeebe1), full GPU offload, three repetitions each, run on
the same idle RTX 5090 for this paper. They measure raw prefill and decode
inside llama.cpp, with no HTTP, no template, no JSON parsing, so they flatter
the generative side.

| Measurement | Time | Rate |
| --- | ---: | ---: |
| Prefill 256 tokens | 33.5 ms | 7,656 tok/s |
| Prefill 1,024 tokens | 85.1 ms | 12,042 tok/s |
| Decode 32 tokens | 143.2 ms | 223.6 tok/s |
| Decode 128 tokens | 564.6 ms | 226.7 tok/s |

Table 2. llama-bench, Gemma 4 26B-A4B UD-Q4_K_XL, RTX 5090, batch 2048. Decode
is 4.4 ms per token regardless of how many.

So the arithmetic is simple. A generative classifier pays the same prefill
Unridden pays, then 4.4 ms per output token. The smallest possible generative
answer, a bare option id in JSON, is around 8 to 12 tokens: 35 to 55 ms on top
of prefill, roughly doubling the latency of a short question. A one-line
justification (32 tokens) triples it. Anything resembling chain-of-thought (128
tokens) is more than ten times the cost of the readout for the same prefill.

| Shape | Unridden, measured | Generative, minimal JSON (est.) | Generative, 32-token answer (est.) |
| --- | ---: | ---: | ---: |
| 1 question, 111-token prompt | 33 ms | about 70 to 90 ms | about 175 ms |
| 1 question over a 1,007-token state | 151 ms | about 130 ms | about 230 ms |
| 8 questions over that state, sequential with prefix reuse | 399 ms | about 1.0 s (8 calls) or about 450 ms (one call, 80-token JSON) | about 1.8 s (8 calls) |
| 8 questions over that state, batched mode | 278 ms | as above | as above |

Table 3. Unridden rows are harness medians (native-v8 sequential, native-v9
batched). Generative rows are prefill plus decode from Table 2, with no
parsing, retries, or template overhead added. The one-call generative variant
returns eight answers in one JSON blob, so it has no per-question distribution
and fails as a unit if the JSON is malformed.

The second row is the interesting one. On a single question over a long state,
the readout is not faster than a minimal generative answer; the 151 ms is
dominated by the 1,007-token prefill either way, and Unridden spends a little
extra on its own overhead. The gap opens with question count and with answer
length, and it opens fast. Ask eight things about one document and the readout
is two to six times cheaper than the cheapest generative alternative, while
returning eight independent distributions instead of one parsed blob.

Latency is the smaller half of the argument. The larger half is what comes
back. A generative classifier returns a string. To get a probability you either
sample many times (multiplying the cost) or ask for logprobs of a constrained
output, at which point you have rebuilt this project inside a chat endpoint
with less control over the readout position. Unridden returns the distribution
on every call, deterministically, with the prompt hash and the label-token
mapping beside it.

The earlier proof of concept went one step further and read a trained linear
head at Gemma block 18 of 30, exiting early when confident: median 17 ms
against 28 ms for the full pass on a three-class banking task, at 114 of 120
accuracy. That path needs a trained head per task and was set aside for the
general API, but it is the natural next lever if the fixed 30 ms per question
ever matters.

![The earlier proof of concept's early-exit cascade: blocks 1 to 18 and a trained head answer when the top probability is at least 0.71, otherwise a full-depth pass reads the label logits](images/early-exit-cascade.svg)

## Determinism and its limits

This is the finding I did not expect and the one worth remembering. For a fixed
configuration the API is exactly deterministic: repeat a request and the raw
logits come back bit-identical, every time, in every run. But the configuration
includes the GPU batch shape, and the model is a Q4 mixture of experts. Two
control runs with prefix reuse switched off, differing only in prefill batch
size (256 against 128), changed 6 of 1,286 answers and moved one probability by
0.82. Turning prefix reuse on, which changes the chunking, changed 4 answers
with a maximum move of 0.59. Moving llama.cpp from afeebe1 to v0.4.1, which
changes the kernels, changed 13, net +3 correct. Batched mode changed 11.

The mechanism, as far as it has been isolated: different batch shapes pick
different kernels, a tiny numeric difference flips which experts fire for a
borderline token, and on questions that were already near a decision boundary
the logits move by several units. Accuracy does not move outside noise (1188 to
1197 across all configurations). But about 1% of these questions are close
enough to a boundary that any change to the numerics will move them. The
practical rule is that a comparison between runs must hold the configuration
fixed, and a stored answer is reproducible only with the recorded worker,
runtime, and settings. That is exactly why the provenance hashes are in every
response.

## Prefix reuse and batched mode

Version 1 cleared the cache and re-prefilled the full prompt for every
question, which made the isolation proof trivial and the latency linear in
question count. Prefix reuse (decision 0002) prefills the shared
instruction-plus-state once per request and trims the cache back to it for
each remainder. On the seven suites the median request dropped from 247 to 194
ms, p95 from 723 to 521 ms, tokens prefilled from 353k to 247k, with the
isolation harness still at exactly 0.0. Eight questions over a 1,007-token
state now take 399 ms; the pre-reuse figure of about 1.1 s is extrapolated from
v1's measured linear scaling, since v1 never ran that shape. Nothing is kept
between requests, so no caller's state ever sits in the cache for the next one.

Batched mode (decision 0004, opt-in, off by default) goes further: after the shared
prefix is in the cache it is copied to one KV sequence per question and all the
remainders decode together in one batch, each question's logits read at its
own last token. That halves the fixed per-question cost (34 to 18 ms over a
long state) and takes the suite median to 154 ms. The price is that a
question's numerics now depend on how many siblings it was sent with. Repeats
and id renames stay at exactly 0.0; solo and reordered comparisons move by a
few parts in ten million in probability. Eleven of 1,286 answers changed and
accuracy went to 1188. A contamination probe, where a sibling in the same batch
states the opposite answer, moved the target's logits by exactly nothing
compared with a neutral sibling of the same length, in both orderings, so batch
shape moves logits and sibling content does not. I left the default sequential.
Batched is for throughput work where one caller controls the request shape and
does not compare answers across requests.

## Limitations and threats to validity

- Twenty-six options, 2,048 prompt tokens, 32 questions per request. Larger
  option sets need a caller-side shortlist stage.
- Confidence is the maximum label probability, uncalibrated and saturated near
  1.0. Do not gate decisions on it without fitting a temperature first.
- The suites are synthetic and written by Claude agents. They were written
  blind and labels were frozen before inference, but two items were revised
  after seeing results; the revised run moved overall accuracy by 0.003.
- 241 cases cannot resolve overall differences under roughly four points or
  per-suite differences under five.
- The generative comparison uses llama-bench, which omits HTTP, templating, and
  parsing. Every generative figure in Table 3 is arithmetic from Table 2, not
  an end-to-end measurement, and it is a floor.
- One model, one quantization, one GPU. Each llama.cpp revision was measured
  once.
- The isolation guarantee is per configuration. A different batch size, reuse
  setting, or kernel revision will move about 1% of borderline answers.
- A question cannot see its siblings. A rubric that relies on a definition in
  another question will score badly, and one suite did.

## Status

The public repository is at github.com/potto007/unridden under Apache-2.0,
with the package, the native worker, the validation harness, the seven suites,
the CLI, the results documents, and community files mirrored from
TrustedCourier (DCO sign-off, code of conduct, security policy, CI on Python
3.12 and 3.13 plus a C++ helper test). Model weights are not included; the base
runtime script builds llama.cpp at the tested release, v0.4.1, and warns rather
than refuses on other revisions. Main
is protected: PRs only, required checks plus DCO. Prefix reuse is in; batched
mode is being ported as the next PR. There is no systemd unit for standing use
yet; every measurement here came from in-process ASGI runs and the batch CLI.

What I would do next, in order: fit a temperature on held-out data and publish
the calibrated confidence beside the raw one; measure repeat runs per
configuration so the 1% boundary movers can be separated from noise; add a
caller-side option to trade the 26-option cap for a two-stage shortlist; and
then decide whether the early-exit head is worth reviving for the
highest-volume tasks.

## Evidence

| Item | Source |
| --- | --- |
| Model | Gemma 4 26B-A4B instruct, Unsloth UD-Q4_K_XL GGUF, sha256 ef728c8e...36e30c |
| Hardware | One RTX 5090 (32 GiB), shared with a resident embedding server; worker peak about 17.6 GiB |
| Runtime | llama.cpp afeebe1 (all private-tree runs, llama-bench), v0.4.1 (public repo revalidation) |
| API validation | gpu-validation-v6 and v8 (134 requests, 36/36 smoke, 72 isolation comparisons at 0.0); failure-paths-v2; negative-control-run-v1 |
| Suites | usecase-suites-v1 (first frozen run), v2 (revised), v4 (prefix reuse), v5 controls, v9 (batched); v0.4.1 revalidation in the public repo |
| Generative | llama-bench, three repetitions per point, run for this paper |
| Decisions | docs/decisions 0001 (full-depth readout), 0002 (prefix reuse), 0003 (tested llama.cpp default), 0004 (batched mode) |

All numbers are taken from the recorded evidence files named above; nothing
was re-derived from memory.
