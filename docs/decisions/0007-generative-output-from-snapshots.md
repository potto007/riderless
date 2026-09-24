# 0007. Generative output from a snapshot, as an experimental /v2 exception

- Status: accepted
- Date: 2026-09-23
- Extends [0006](0006-split-execution-for-state-snapshots.md). Makes one
  scoped exception to the zero-generated-tokens rule; `/v1` and every other
  `/v2` route are unchanged.

## Context

Unridden answers by reading label logits and never samples. The state-snapshot
design put text generation "outside this design". Measured on 2026-09-23, a
caller that has already snapshotted a state for decisions often wants a
generated artifact from that same state next (a drafted reply that respects
the routing and urgency answers). Re-prefilling the state elsewhere throws away
the snapshot's work. Splicing the two layer-range K/V halves into a stock
llama.cpp context does work, but it depends on llama.cpp's private KV
serialization layout, needs a hand-rewritten sequence-file version, only
supports block-30 snapshots, and is not bit-identical to the split graph that
produced the state.

## Decision

- **Generate on the split runtime, from the snapshot.** The snapshot worker
  gains a `generate` command: restore the parent's lower (blocks 0-17) and
  upper (18-29) sequence states, prefill only the suffix, then loop one token
  at a time through lower, H18 into upper via `llama_batch.embd`, head, sampler.
  The snapshot prefix is never recomputed, and the worker trims back to the
  parent afterwards like every other branch.
- **Exposed as `POST /v2/outputs`,** registered only with `--snapshots`, so it
  is off by default and experimental with the rest of `/v2`. Responses report
  `output_tokens` truthfully; the `/v1` `output_tokens: 0` contract and every
  decision route's zero-token accounting are unchanged.
- **Greedy decoding, bounded.** `max_tokens` is at most 1024 and prompt plus
  `max_tokens` must fit the context, or the request is refused before
  inference with the existing budget error.
- **Honest baselines.** Behind `--reference` the worker can also generate by
  re-prefilling the same prompt on the split graph (`split_prefill`) or on the
  stock graph (`reference`), with a `prefill_break` so the chunk boundary
  matches the snapshot path. Chunk shape alone moves logits by about 0.8 on
  this model, so an unmatched baseline is not a fair comparison.

## Consequences

- Measured on an RTX 5090 with a 717-token snapshot and a 145-token suffix:
  warm time to first token 28-49 ms against 125-140 ms for a full re-prefill,
  about 180-200 tokens/s, and logits bit-identical on all 285 steps to a
  split-graph re-prefill broken at the same token.
- Output inherits the split-vs-stock drift of 0006 (gate 5 still fails): about
  0.8 logit at step 0 against the stock graph, with greedy text diverging after
  roughly 150 tokens. Both remain coherent; neither is the reference.
- The zero-generated-tokens claim now holds for `/v1` and the `/v2` decision
  routes only. The README documents `/v1` alone and is unchanged; any future
  `/v2` documentation must name this exception.
