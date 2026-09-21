# Batched question evaluation

Run on 2026-09-21, one RTX 5090, Gemma 4 26B-A4B UD-Q4_K_XL. Decision record:
[0004](../decisions/0004-optional-batched-question-evaluation.md).

**Every number on this page was measured on llama.cpp release `v0.4.1`, commit
`b29c606e28a01b1bc8c1351026a0fa6e616bf6c4`**, with a worker built from this
branch's `riderless/api/native/worker.cpp` against that base runtime. All four
runs below use the same worker binary; the only thing that differs is the
`--batched` flag.

| Run | Mode | Command |
| --- | --- | --- |
| Validation, sequential | default | `validate_api_v1.py --allow-gpu` |
| Validation, batched | `--batched` | `validate_api_v1.py --allow-gpu --batched` |
| Suites, sequential | default | `run_usecase_suites.py --allow-gpu` |
| Suites, batched | `--batched` | `run_usecase_suites.py --allow-gpu --batched` |

The run directories live under `outputs/`, which this repository does not
commit. The two comparison scripts named below regenerate every figure from a
pair of those directories.

## What was built

An opt-in mode that gives every question of a request its own KV sequence over
one shared prefix and evaluates all the remainders together.

- The shared prefix is prefilled once into sequence 0, in the same chunks the
  sequential path uses, then `llama_memory_seq_cp` copies it to one sequence per
  question. With a unified cache that copy only tags the existing cells with
  another sequence id, so it costs no cells and no buffer copy.
- Every question's remainder is appended to one token list with its own sequence
  id and its own positions, and the list is decoded in `ceil(total / n_batch)`
  `llama_decode` calls. Each question reads its logits at its own last token,
  from the decode that contained it.
- It is a worker start-up flag (`ApiConfig.batched`, CLI and runner `--batched`,
  worker `--batched --batched-context N`), because `n_seq_max`, `kv_unified` and
  `n_ctx` are context construction parameters. The default worker is unchanged.
- Worker protocol version 3. Every question reports `evaluation_mode` and
  `batch_sequences`; the request reports `batched_fallback`, which the API
  surfaces to callers as `evaluation: {mode, fallback}` in the response body,
  not only under `?diagnostics=true`. The handshake carries `batched_mode` and
  `batched_context`, and `schema.py` plus `mapping.py` reject a report that
  contradicts them.

## The default mode is untouched

The seven suites on the new worker with the flag off, against the recorded
v0.4.1 suite run from the previous change
([llama-v0.4.1-revalidation.md](llama-v0.4.1-revalidation.md)):

| | Result |
| --- | --- |
| Questions compared | 1,286 |
| Raw label logits identical | yes, bit for bit |
| Answers changed | 0 |
| Largest probability move | 0.0 |
| Accuracy | 1197 of 1286 in both |
| Tokens prefilled | 246,981, unchanged |

Sequential responses carry no `evaluation` field at all, so the default body is
byte for byte what it was.

The validation harness in sequential mode reproduced the earlier result: 36 of
36 smoke answers, 0 generated tokens, 141 requests of which 10 are expected
rejections, and all 72 solo, reversed, renamed and repeated comparisons differ
by exactly 0.0 in probability and in raw logits.

## The seven suites in batched mode

Same 241 requests and 1,286 questions, batched against sequential on the same
worker:

| | Sequential | Batched |
| --- | --- | --- |
| Questions correct | 1197 of 1286 (93.08%) | 1192 of 1286 (92.69%) |
| Rankings, top-1 | 32 of 32 | 31 of 32 |
| Rankings, pairwise concordance | 120 of 120 | 119 of 120 |
| Answers changed | reference | 9 |
| Questions moved over 0.3 | reference | 10 |
| Largest probability move | reference | 0.954 |
| Largest raw logit move | reference | 6.93 |
| Median request | 182.9 ms | 153.1 ms |
| p95 request | 512.0 ms | 456.0 ms |
| Total | 53.7 s | 45.2 s |
| Tokens prefilled | 246,981 | 246,981 |
| Sequential fallbacks | not applicable | 0 of 241 |

Every one of the 1,286 questions was evaluated in batched mode; the largest
request put 13 sequences in one batch, and the distribution of sibling counts
runs 1 (7 questions), 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13 (39 questions).
Nothing was tuned against these suites: the only knob the run sets is
`--batched`, and `batched_context` stayed at its 8192 default.

The 9 changed answers are 5 in the guardrail suite, 2 in candidate-span, and one
each in claim-support and hierarchical classification. The guardrail suite is the
one earlier audits already flagged as noisy, and it is where the batch-size
control run in [prefix-reuse.md](prefix-reuse.md) also concentrated its changes.

The rankings regression is the same suite and a single case:
`guardrail-policy-gates` / `pii-coworker-home-address`, hazard battery
`hazard-battery-by-probability`. In batched mode `hz_illicit` (0.9999906) ranks
just above the expected `hz_pii` (0.9999900), which costs the top-1 hit and one
of that case's five pairs. The sequential run's failed-ranking list is empty.
Both hazards sit within 1e-05 of 1.0, so this is the same boundary-question
drift as the 9 changed answers rather than a separate failure mode, but it is
the one metric where batched mode gives up a previously perfect score.

## Determinism

Within the batched run, the 12 repeats of an identical request in the validation
harness differ by exactly 0.0 in probability and in raw logits, and the harness
asserts that. The suites themselves were run once per configuration here, so
this page does not repeat the earlier finding that two independent suite runs of
one configuration are bit-identical; it only shows that the sequential
configuration reproduces its own recorded run bit for bit.

That leaves the 5-question accuracy gap unresolved between "a real property of
this configuration on this suite" and "borderline questions landing differently".
It is a small enough margin that another corpus could put batched mode ahead.

## Scaling

Validation-harness benchmark axes, median of three repeats, API round trip:

| Request shape | Sequential | Batched |
| --- | --- | --- |
| 1 question, 111-token prompt | 27.2 ms | 32.1 ms |
| 2 questions | 51.6 ms | 39.3 ms |
| 4 questions | 102.1 ms | 77.0 ms |
| 8 questions | 200.5 ms | 153.7 ms |
| 1 question over a 1,007-token state | 150.8 ms | 152.0 ms |
| 2 questions over that state | 179.5 ms | 171.4 ms |
| 4 questions | 233.7 ms | 204.3 ms |
| 8 questions | 339.6 ms | 278.9 ms |

Each extra question over the long state costs about 27.0 ms sequentially and
about 18.1 ms batched: batching removes about a third of the fixed per-question
cost, not most of it. A single-question request is unchanged, as it must be: its
batch of one is the same work.

## What batching costs: sibling independence

Sequential mode asserts that a question is computed identically alone,
reordered, or beside any siblings. Batched mode cannot, so the harness records
the same comparisons instead of asserting them:

| Variant | n | Median probability delta | Max probability delta | Median raw logit delta | Max raw logit delta |
| --- | --- | --- | --- | --- | --- |
| solo (question alone vs in its request) | 36 | 1.7e-08 | 1.9e-06 | 0.44 | 1.41 |
| reversed (question order flipped) | 12 | 5.5e-08 | 1.2e-06 | 0.70 | 1.17 |
| renamed (question ids changed) | 12 | 0.0 | 0.0 | 0.0 | 0.0 |
| repeat (identical request again) | 12 | 0.0 | 0.0 | 0.0 | 0.0 |

A repeat of the identical request is still exactly 0.0 and is still asserted in
both modes, as is the renamed variant's result even though it is only recorded.

The raw logit deltas in the solo and reversed rows are much larger than the
probability deltas, because the shift is close to a constant offset across the
label logits and the softmax over the label alphabet removes it. On these 12
fixture cases no answer flipped. The suites, which are 100 times larger and
include borderline questions, are where the flips show up: 9 of 1,286.

## Cross-question contamination: measured and asserted

The design fixes the batch shape and varies only content the question must not
be able to see. Two-question requests over one state that says the flag is
green, asking "Choose the explicitly stated flag color":

- sibling repeats "The wall is red." 24 times (irrelevant),
- sibling repeats "The door is red." 24 times (irrelevant),
- sibling repeats "The flag is red." 24 times (contradicts the state and names
  the answer the question must not give).

The set runs in both question orders: with the target question first (target in
sequence 0, its tokens occupy the earliest batch slots) and with the target
question last (the sibling's remainder precedes the target's in the same batch).
The harness asserts that all three probes of an order have identical
per-question token counts, so batch composition, ubatch split and every KV
position are identical and only the bytes in the sibling's own cells differ.

Result, in both modes and both orders: the target question's raw label logits
and probabilities are bit-identical across the three siblings. Noise floor delta
0.0, adversarial delta 0.0, and the target answered "green" in every probe.

This is a check, not only a measurement. The harness asserts, per order, that
the adversarial swap moves the target no further than the neutral swap does
(`delta <= max(noise_floor_delta, 1e-6)` for probabilities and for raw logits)
and that the target still answers "green" in all three probes. A run where a
sibling's content moved the target would fail instead of printing
`API_VALIDATION_COMPLETE`.

Why this would catch a wrong attention mask: the "wall"/"door" pair bounds
everything that could move the question for reasons other than leakage, given an
identical batch shape, and that bound measured 0.0. The "flag" sibling is a
maximal semantic perturbation of the same length: 24 assertions that the flag is
red, against a state that mentions green twice. If a question's tokens could
attend to a sibling's remainder cells, that content would have to move the
green/red logits, and any such move exceeds a floor of exactly zero. Running
both orders is what covers a defect that leaks only in one direction, for
instance a mask that unmasked every cell preceding the query in batch order
regardless of sequence id: with the target first, its tokens precede the
sibling's and such a defect would be invisible. Two limits remain: a mask bug
that leaked only the prefix would not be caught, but the prefix is identical for
both questions by construction, so there is nothing there to leak; and the probe
is two questions wide, so it does not exercise leakage between two siblings that
are both non-first.

The worker also checks, after every batched request, that each sequence's
maximum KV position equals its own prompt length minus one. That catches a
sequence that gained cells beyond its own last position; it would not catch
extra cells at positions below it, which is why the content probe above exists.

The mechanism behind the result, read from the tested llama.cpp revision: with
`kv_unified = true` the cache has one stream, so `llama_kv_cache::seq_cp` only
adds the destination sequence id to the prefix cells, and
`set_input_kq_mask_impl` drops every cell that does not carry the querying
token's sequence id (both in `src/llama-kv-cache.cpp`). A remainder token is
written with exactly one sequence id, its own. Gemma 4's sliding-window layers
use `llama_kv_cache_iswa`, whose `seq_cp` forwards to both the base and the SWA
cache, and `swa_full = true` makes the SWA cache the same size as the base one,
so no part of a copied prefix is dropped.

## The sequential fallback, exercised on the GPU

A request that needs more KV cells than `batched_context` is answered by the
0002 sequential algorithm inside the batched context. The harness sends one such
request in both modes: 6 questions of 1,817 tokens each (each well
under the 2048-token per-question limit) over a shared state, needing 9,132
cells against the 8,192 available.

- In batched mode the response carries
  `evaluation: {"mode": "sequential", "fallback": "context"}`, every question
  reports `evaluation_mode = "sequential"` and `batch_sequences = 1`, and
  nothing is truncated.
- In sequential mode the same request is answered normally and carries no
  `evaluation` field.

Comparing the two: the fallback path is **not** bit-identical to the sequential
worker. Largest raw label logit move 0.0238, largest probability move 3.3e-08,
and all 6 answers identical ("green"). Those three numbers come from
`scripts/riderless/compare_fallback.py` reading the two `summary.json` records,
so a rerun that moves them can be diffed rather than recomputed by hand. That is
the expected size of a same-algorithm, different-context-shape difference, and it
confirms that the fallback is a third numeric regime rather than a copy of the
default worker's output. One request is not a general claim about how far the two
can diverge.

## VRAM

| | Sequential | Batched (8192 cells) |
| --- | --- | --- |
| `n_ctx` / `n_seq_max` / `kv_unified` | 2048 / 1 / false | 8192 / 32 / true |
| Non-SWA KV buffer | 40 MiB | 160 MiB |
| SWA KV buffer | 400 MiB | 1600 MiB |
| CUDA0 compute buffer | 268.3 MiB | 266.5 MiB |
| Host compute buffer | 7.5 MiB | 13.5 MiB |
| Whole-card peak during the run | 22,892 MiB | 24,182 MiB |
| Free on a 32,607 MiB card at peak | 9,715 MiB | 8,425 MiB |

The card's idle baseline during these runs was about 5.2 GiB of other resident
work. Batched mode costs 1,320 MiB of KV, about 0.21 MiB per cell, and leaves
8.2 GiB free. With `batched` on, `ApiConfig` refuses a `batched_context` above
`max_questions * context_size` (65,536 cells at the defaults, about 13.4 GiB of
KV), which is the most any request could need. Both `batched_context` bounds are
gated on the flag, because a sequential worker allocates no batched cache and
must not be refused over an unused default.

These peaks are from the validation harness, which samples `nvidia-smi` once a
second. The suite runner does not sample the GPU, so there is no separate peak
for the suite runs; they use the same worker and the same context, and their
requests are no larger than the harness's.

## Is the speedup worth the loss of sibling independence?

For the suite workload, honestly: only sometimes.

- The gain is 16% off the median request and 16% off the wall clock of the whole
  suite run, measured against the sequential run on the same worker (182.9 to
  153.1 ms, 53.7 to 45.2 s). It is much larger for many short questions over one
  state (8 questions: 201 to 154 ms, and 340 to 279 ms over a long state) and
  zero for a single question.
- The price is that a question's result depends on how many siblings the caller
  happened to send and how long they are. Two callers asking the same question
  over the same state get different numbers if their other questions differ, and
  9 of 1,286 suite answers changed, with one probability moving 0.95.
- Accuracy fell from 1197 to 1192 of 1286. Five questions on one hand-authored
  corpus is too thin to call a general accuracy loss, and this page did not
  repeat either suite run.

The recommendation that follows: leave the default alone. Turn batched mode on
for throughput work where the caller controls the request shape and a sub-1%
boundary-question drift is acceptable, for example bulk scoring of a fixed
question set. Do not turn it on for a service whose callers compose requests
independently and compare results, because there the sibling dependence is a
correctness surprise, not a performance trade.

## Not done

- Batched mode was measured at `batched_context = 8192` and `n_batch = 256`
  only. Other cache sizes and batch sizes were not swept, so nothing says
  whether a different batch size recovers the 5 lost answers.
- One suite run per configuration. Repeat determinism is shown only by the 12
  identical-request repeats in the validation harness, not by a second full
  suite run.
- The fallback comparison is one request. It shows the fallback path differs
  from the default worker; it does not bound how far.
- The contamination probe is two questions wide and uses one target question
  over one state. It does not exercise leakage between two non-first siblings,
  and it is a single semantic contrast rather than a family of them.
- The worker's protocol-level rejection paths for batched mode were not fuzzed,
  and `validate_api_failures.py` was not re-run against the batched worker.
- The suite runner does not sample `nvidia-smi`, so the VRAM peaks come from the
  validation runs, not the suite runs.
