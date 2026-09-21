# Comparing two worker builds: an added invariant is answer-neutral

Written on 2026-09-21 from evidence already on disk. No GPU work and no rerun
was performed for this report; every number is read from an existing artifact.

Two consecutive builds of the non-generative worker differ by one added runtime
invariant check. Against the same fixture, the same model, and the same
122-request validation script they produced numerically identical answers,
probabilities, and raw logits. The only fields that differ between their
evidence files are latency measurements, and those differences are inside
run-to-run noise for the sample sizes involved.

## What changed

Both builds share the same llama.cpp revision, the same runtime bundle, the same
model, and the same runtime configuration (context 2048, batch and ubatch 256,
8 threads, causal attention, `swa_full`, CUDA fusion and graphs on, callbacks
off, 0 generated tokens, `execution_mode: full`). The only differing entry in
`build.json` is the source hash of `worker.cpp` and, consequently, the
executable hash.

The delta is four lines in `evaluate_question`, immediately after the existing
`llama_memory_clear`:

```cpp
// cache_cleared below is a measured fact: sequence 0 must hold no positions.
if (llama_memory_seq_pos_max(llama_get_memory(context), 0) != -1) {
    throw std::runtime_error("Context memory was not empty before prefill");
}
```

That turns a claimed invariant (the KV cache is empty before each question's
prefill) into a checked one. The shipped worker generalizes the same check for
shared-prefix reuse: it compares the held position against `held`, which is
`prefix - 1` when a prefix is reused and -1 otherwise, so it now asserts the
reused prefix exactly. See `evaluate_question` in
`riderless/api/native/worker.cpp`. The preceding build step had introduced a
`budget_error` exception type so that an over-context prompt reports
`reason: "budget"` instead of a generic preflight reason.

## Outcome 1: the check is answer-neutral

Both runs used the same frozen fixture, the same 36 questions, and the same 122
recorded requests, one model load each. Diffing the two summary files key by key
gives 156 differing leaves, and every one is a latency measurement. No other
leaf differs:

| Section | Older build | Newer build | Identical |
| --- | --- | --- | --- |
| Semantic smoke (36 questions: 17 choice, 5 score, 14 noul) | 36/36 correct | 36/36 correct | yes, including every target probability and NLL to full precision |
| Isolation (72 comparisons) | max probability delta 0.0, max raw logit delta 0.0 | same | yes |
| Requests | 122 | 122 | yes |
| Rejected requests | identical | identical | yes |
| Generated tokens | 0 | 0 | yes |
| Executed input tokens | 49,670 | 49,670 | yes |

The added invariant fired on no request in this fixture and changed no readout.
That is the result you want from an assertion: it either holds silently or it
stops the run. It held.

## Outcome 2: the latency cost is not separable from noise

Aggregate API latency over the validation run:

| Statistic | Older build | Newer build | Delta |
| --- | --- | --- | --- |
| mean | 92.04 ms | 93.19 ms | +1.25% |
| median | 96.51 ms | 97.05 ms | +0.56% |
| p95 | 246.75 ms | 243.29 ms | -1.40% |

Per benchmark group (each group is n = 3 requests in a single run):

| Axis | Size | Input tokens | Older API median | Newer API median | Delta |
| --- | --- | --- | --- | --- | --- |
| questions | 1 | 111 | 31.98 ms | 34.43 ms | +7.66% |
| questions | 2 | 222 | 64.69 ms | 68.10 ms | +5.28% |
| questions | 4 | 444 | 126.25 ms | 130.19 ms | +3.12% |
| questions | 8 | 888 | 254.58 ms | 259.61 ms | +1.97% |
| padding words | 0 | 111 | 32.11 ms | 33.69 ms | +4.92% |
| padding words | 384 | 495 | 74.78 ms | 75.09 ms | +0.41% |
| padding words | 896 | 1007 | 133.91 ms | 136.58 ms | +1.99% |
| padding words | 1728 | 1839 | 244.01 ms | 242.00 ms | -0.83% |

Read this carefully rather than as a regression. The added work is one
`llama_memory_seq_pos_max` call per question, which cannot plausibly cost the
2.43 ms that the single-question bucket shows. Each bucket is three requests
from one process with its own model load, and one bucket moves the other way.
The honest statement is that no per-question overhead attributable to the check
is measurable at this sample size, and the observed spread is run-to-run
variance. Isolating a sub-millisecond cost would need repeated interleaved runs
of both binaries, which was not done.

Worker startup: 9.58 s versus 9.71 s, one observation each.

## Outcome 3: a second confirmation from the suite runs

Comparing the two use-case suite runs question by question, 1,283 of the 1,286
answers are byte-identical across the two builds. The three that differ are all
in `find-overlapping-room-booking`, whose query text was rewritten, so its input
changed and a different answer is expected. A fourth changed outcome,
`pii-bulk-export-buried/ctx_third_party`, has an identical model answer under
both builds; only the gold label flipped.

That comparison spans two build steps and a gold-label revision, because the
worker change and two suite files moved in the same commit.

## What this report cannot say

- Nothing about whether the invariant ever fires. It did not fire in 122
  validation requests or 241 suite cases, which bounds its trigger rate on these
  workloads at zero observations, not at zero.
- No isolated quality comparison for the intermediate build exists, and none can
  be produced from the artifacts: that build never ran the suites.
- No per-question cost for the added check. See Outcome 2.
- Nothing about the `budget` error-reason split beyond its source diff. The
  rejected-request section is byte-identical between the two runs, so it does not
  separate the builds either.
