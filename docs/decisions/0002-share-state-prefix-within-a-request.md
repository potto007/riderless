# 0002. Prefill the shared state once per request and reuse it across that request's questions

- Status: accepted
- Date: 2026-09-21
- Supersedes: the "fresh context per question" part of
  [0001](0001-full-depth-label-readout-in-an-owned-child.md). The rest of 0001
  stands.
- Note: renumbered from an earlier internal record, with the machine-specific
  context removed.

## Context

0001 cleared the context and prefilled the full prompt for every question, and
deferred prefix sharing until it had its own isolation proof. Requests in the
use-case suites average five questions over one state, so most prefill work was
repeated: 353k tokens prefilled where about 247k were distinct.

## Decision

Within one request the worker prefills the text shared by every question (the
system instruction and the state) once. Later questions trim the context back to
that prefix with `llama_memory_seq_rm` and prefill only their remainder.

- The split point is a function of the question's own prompt, never of its
  siblings, and the prefix is reused only on exact token equality. A question is
  therefore computed identically alone, reordered, or beside any siblings.
- A shared prefix under 128 tokens is not split.
- The cached prefix never outlives the request. No state crosses callers.
- The compiler marks the shared text with `shared_prefix_bytes`. It is a hint:
  any split below the prompt length is correct, and the prompt is unchanged.
- Diagnostics report `prompt_tokens`, `processed_tokens`, `reused_tokens`, and
  `cache_cleared` as measured facts. Worker protocol version 2.
- `ApiConfig.share_prefix=False` restores the 0001 behaviour bit for bit.

## Alternatives rejected

- **Longest common token prefix across the request's questions.** Simpler, but
  the split would depend on the siblings, so a question's chunking and numbers
  would change with its neighbours.
- **Keep the prefix across requests.** Helps repeated states, but one caller's
  state would persist in the context for the next caller.
- **Evaluate all questions in one decode pass.** Faster still, but it makes a
  result depend on the number of siblings through the batch-shape sensitivity
  measured in [the prefix-reuse results](../results/prefix-reuse.md), which is
  exactly the isolation property 0001 was built to keep. Left as a possible
  opt-in mode, not built.

## Consequences

- Seven use-case suites: median request 247 to 194 ms, p95 723 to 521 ms, tokens
  prefilled 353k to 247k, accuracy 1193 to 1194 of 1286. Eight questions over a
  1,007-token state: about 1,100 ms to 399 ms, where the no-reuse figure is
  extrapolated from measured linear scaling. Short states are unchanged; a single
  question over a long state pays about 14 ms for the split.
- The isolation proof holds unchanged: 72 comparisons differ by exactly 0.0.
- Turning reuse on changes the prefill batch shapes, and about 1% of borderline
  questions move, as they also do when only the batch size changes with reuse
  off. Results are deterministic per configuration, not across configurations.
- Evidence: [shared-prefix reuse results](../results/prefix-reuse.md).
