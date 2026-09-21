# Shared-prefix reuse

Run on 2026-09-21, one RTX 5090, Gemma 4 26B-A4B UD-Q4_K_XL. Decision record:
[0002](../decisions/0002-share-state-prefix-within-a-request.md).

**Every number on this page was measured on llama.cpp revision
`afeebe103bd99cda8f5dfaefcabadf890db7fda7`.** The tested revision is now
release `v0.4.1`; see
[llama-v0.4.1-revalidation.md](llama-v0.4.1-revalidation.md).

## What changed

Every question of a request starts with the same text: the system instruction
and the state. The earlier worker cleared the context and prefilled that text
again for every question. The current worker prefills it once for the first
question, and each later question trims the context back to it and prefills only
its own remainder. Nothing is kept between requests.

Three rules keep a question's result independent of its siblings:

- The split point is computed from the question's own prompt only.
- The prefix is reused only when the cached tokens equal it exactly, so it is
  always prefilled in the same chunks.
- A shared prefix under 128 tokens is not split at all. The split costs one
  extra decode (about 14 ms), which a short prefix does not repay.

Diagnostics carry `prompt_tokens`, `processed_tokens`, and `reused_tokens`
(`processed + reused == prompt`), and `cache_cleared` is true exactly when
nothing was reused. `usage.input_tokens` counts tokens actually prefilled. The
worker protocol is version 2. `ApiConfig.share_prefix=False` restores the old
behaviour.

## Validation harness

134 requests, 36 of 36 smoke answers, 0 generated tokens. All 72 mixed, solo,
reversed, renamed, and repeated comparisons differ by exactly 0.0 in probability
and in raw logits, the same result as before reuse. 37,050 tokens were reused
against 57,935 prefilled.

| Request shape | Without reuse | With reuse |
| --- | --- | --- |
| 1 question, 111 tokens | 34 ms | 33 ms |
| 8 questions, 111 tokens each (short state, no split) | 260 ms | 250 ms |
| 1 question over a 1,007-token state | 137 ms | 151 ms |
| 2 questions over that state | about 274 ms (2 x 137) | 187 ms |
| 4 questions | about 548 ms | 256 ms |
| 8 questions | about 1,096 ms | 399 ms |

The no-reuse figures for several questions over the long state are extrapolated
from its measured linear scaling; that axis was not run directly without reuse.
Each extra question over the long state now costs about 35 ms, which is the
fixed cost of one decode and no longer depends on the state length.

## The seven use-case suites

Same 241 requests and 1,286 questions as the revised suite run in
[usecase-suites.md](usecase-suites.md).

| | Without reuse | With reuse |
| --- | --- | --- |
| Questions correct | 1193 | 1194 |
| Median request | 247 ms | 194 ms |
| p95 request | 723 ms | 521 ms |
| Total | 69.6 s | 55.2 s |
| Tokens prefilled | 353,251 | 246,997 |

## Answers are sensitive to the prefill batch shape

Reuse changed 4 of 1,286 answers and moved one probability by 0.59. That is not
caused by reuse as such. Two control runs on the same worker with reuse off show
it:

| Run | Identical answers to the reference run | Largest probability change | Correct |
| --- | --- | --- | --- |
| Reuse off, batch 256 | 1286 of 1286, bit for bit | 0.0 | 1193 |
| Reuse off, batch 128 | 6 answers changed; 8 questions moved over 0.3 | 0.82 | 1191 |
| Reuse on, batch 256 | 4 answers changed; 10 questions moved over 0.3 | 0.59 | 1194 |

Changing only the prefill batch size, with no reuse, disturbs more than reuse
does. The likely cause, not isolated here: GPU kernels differ by batch shape, and
in a mixture-of-experts model a tiny numeric difference can flip which experts
run, so a few borderline questions move by several logits. The affected
questions are mostly in the guardrail suite, which the earlier audit already
flagged as noisy, and sit on prompts of 250 to 400 tokens, where the old run used
one 256-token batch and the new one uses two smaller ones.

Consequences:

- For a fixed configuration the API is deterministic and sibling-independent.
  That is what the harness proves, and it still holds.
- Answers are not stable across configurations (batch size, reuse on or off).
  About 1% of these questions sit close enough to a boundary to move. Any
  comparison between runs must hold the configuration fixed.
- Accuracy is unaffected within noise: 1191, 1193, 1194 across the three.

## Not done

- The negative-control build was not repeated for the reuse worker.
- No independent review of the reuse change yet.
- Evaluating all questions in one decode pass would remove most of the remaining
  35 ms per question, at the price of making a result depend on the number of
  siblings through the mechanism described above. Not built.
