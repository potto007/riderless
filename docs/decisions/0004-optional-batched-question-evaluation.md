# 0004. Evaluate a request's questions in one batched decode, as an opt-in worker mode

- Status: accepted
- Date: 2026-09-21
- Extends [0002](0002-share-state-prefix-within-a-request.md). The sequential
  mode 0002 describes stays the default and is unchanged.

## Context

0002 made every question after the first trim the KV cache back to the shared
prefix and prefill only its own remainder. That removed the repeated prefix work
but left a fixed cost per question, which is one `llama_decode` round trip and is
independent of how much text the question adds. A request of eight questions over
one state still pays eight of them.

0002 listed "evaluate all questions in one decode pass" as a rejected
alternative, because a result would then depend on how many siblings the request
carries. The measured fact behind that objection is recorded in
[the prefix-reuse results](../results/prefix-reuse.md): on this Q4
mixture-of-experts model, changing only the prefill batch shape moved a handful
of borderline answers. A mode that puts several questions in one batch changes
batch composition by construction, so it cannot be bit-identical to the
sequential mode.

## Decision

Add a batched evaluation mode that is off by default and is chosen when the
worker starts, not per request.

- **Start-up flag, not a per-request field.** The context has to be created with
  `n_seq_max = max_questions`, `kv_unified = true` and an `n_ctx` large enough
  for the prefix plus every remainder. Those are context construction
  parameters, so a request cannot change them. A per-request switch would also
  mean one process serving two numerically different regimes, which makes any
  measurement ambiguous. The worker is therefore either a sequential worker or a
  batched worker for its whole life. `ApiConfig.batched` (CLI `--batched`,
  runner `--batched`) selects it; `ApiConfig.batched_context` sizes the KV cache
  and defaults to 8192 cells.
- **Default behaviour is untouched.** With the flag off the context parameters,
  the code path, and the decode shapes are exactly those of 0002. The seven
  use-case suites reproduce the recorded v0.4.1 run bit for bit.
- **How a batched request runs.** The shared prefix is prefilled once into
  sequence 0 in the same chunks as before, then `llama_memory_seq_cp` copies it
  to one sequence per question. Every question's remainder is appended to a
  single token list, in request order, with its own sequence id and its own
  positions, and that list is decoded in `ceil(total / n_batch)` `llama_decode`
  calls. Each question's logits are read at its own last token from the decode
  that contained it.
- **The batched split point depends on the siblings.** It is the longest token
  prefix shared by every question of the request, never longer than the split
  each question computed from its own prompt alone, and it is dropped to zero
  below the 128-token minimum of 0002. Sequential mode keeps the
  sibling-independent rule; batched mode has already given up sibling
  independence, so the simpler request-wide split is used there.
- **No usable shared prefix.** When the shared prefix is zero, every sequence
  prefills its whole prompt. Batched mode still applies; only `seq_cp` is
  skipped. Every question then reports `reused_tokens = 0` and
  `cache_cleared = true`.
- **When a batched request does not fit, it falls back and says so.** The
  per-question prompt limit stays the context size (2048 tokens) and is still
  enforced before inference, exactly as today. The batched request additionally
  needs `prefix + sum(remainders)` KV cells. If it needs more than
  `batched_context`, the whole request is evaluated by the 0002 sequential path
  inside the same context, and the worker reports `batched_fallback = "context"`.
  Nothing is truncated and nothing is silently dropped. That path is a third
  numeric regime: the 0002 algorithm, but in the larger unified context, so it
  is not assumed to equal the default worker's output, it is measured against
  it. Every response from a batched worker carries `evaluation: {mode, fallback}`
  in the public body, so a caller that does not ask for diagnostics still learns
  which regime answered it.
- **Honest per-question diagnostics.** Every question reports `evaluation_mode`
  (`sequential` or `batched`) and `batch_sequences` (how many question sequences
  shared its decodes, 1 in sequential mode).
  `processed_tokens + reused_tokens == prompt_tokens` still holds: the prefix is
  charged once, to the first question. `timing_ms` in batched mode is the time
  from the previous question's readout to this question's readout, so the
  per-question values still sum to the request's native time. The worker
  protocol is version 3; the handshake carries `batched_mode` and
  `batched_context`, and the mapping layer rejects any report that contradicts
  them.
- **Cross-question contamination stays impossible, and is measured.** With a
  unified KV cache `seq_cp` only adds a sequence id to the prefix cells, and the
  attention mask drops every cell that does not carry the querying token's
  sequence id (`set_input_kq_mask_impl` in `src/llama-kv-cache.cpp` at the tested
  revision). A sibling's remainder cells never carry another question's sequence
  id, so they are always masked. The worker additionally checks after each
  batched request that each sequence's maximum KV position equals its own prompt
  length minus one, which catches a sequence that gained cells beyond its own
  last position but not extra cells below it. The validation harness measures the
  rest: see "Consequences".

## Alternatives rejected

- **Per-request batched flag.** Needs the large context and the extra sequences
  to exist anyway, so the VRAM cost is paid whether or not a request uses them,
  and one process would produce two numerically different answers for the same
  input. Rejected for measurement clarity.
- **One sequence per question with a non-unified KV cache.** `n_ctx` would be
  divided by `n_seq_max`, the prefix would be physically copied per sequence,
  and the copy would cost both VRAM and a buffer copy. Unified is strictly
  better here because the sequences share a long prefix.
- **Keep the sibling-independent split in batched mode.** Would need a separate
  prefix per question and so a separate prefill per question, which is the cost
  the mode exists to remove.
- **Truncate or reject a request that does not fit the batched context.**
  Truncation is silent data loss. Rejection turns a working request into an
  error for a purely internal reason. Falling back to the sequential path and
  reporting it is the honest option.

## Consequences

- Batched mode is not bit-identical to sequential mode and cannot be. Answers
  move for the same reason a batch-size change moves them.
- A question's result in batched mode depends on the number and the length of
  its siblings, because they set the batch composition. The default mode's
  exact-0.0 isolation guarantee does not hold there, so the validation harness
  stops asserting it in batched mode and records the solo, reversed, and renamed
  deltas instead. A repeat of the identical request must still be exactly 0.0,
  and that stays an assertion in both modes.
- Contamination is checked by a probe that holds the batch shape fixed and
  varies only the content of a sibling's remainder: two-question requests whose
  other question has the same token count but says "The wall is red", "The door
  is red", or "The flag is red", over a state that says the flag is green. The
  first two give the noise floor of swapping content the target question cannot
  see; the third additionally contradicts the state and names the answer the
  target must not give. The harness asserts that the adversarial swap moves the
  target no further than the neutral one, in probabilities and in raw logits,
  and that the target still answers "green". A mask that let a question attend
  to a sibling's remainder would have to move it beyond that floor, because the
  adversarial sibling is a maximal semantic perturbation while the neutral ones
  are not. The set is run in both question orders, so the target is once the
  first sequence in the batch and once the last: a defect that leaks only from
  earlier batch slots into later ones is invisible in one order.
- VRAM: 8192 cells cost 1,760 MiB of KV against 440 MiB at 2048, and
  `batched_context` is capped at `max_questions * context_size`, the most any
  request could need. The measured CUDA0 compute buffer did not grow (268.3 to
  266.5 MiB); the host compute buffer grew from 7.5 to 13.5 MiB. Measured peaks
  are in [the batched-mode results](../results/batched-mode.md).
- Evidence: [the batched-mode results](../results/batched-mode.md).
