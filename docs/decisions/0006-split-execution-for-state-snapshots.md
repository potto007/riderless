# 0006. Split-depth execution in two layer-range contexts for state snapshots

- Status: accepted
- Date: 2026-09-22
- Implements [the state-snapshot design](../superpowers/specs/2026-09-22-state-snapshots-design.md)
  and its [worker protocol](../superpowers/specs/2026-09-22-state-snapshots-protocol.md).
  The v1 worker, its runtime and ADRs 0001-0005 are unchanged.

## Context

A snapshot after block 18 must be resumable: blocks 19-30 run later on the
saved prefix without repeating blocks 1-18. A single ordinary llama.cpp cache
cannot hold tokens that are processed at some layers and not at others, since
its cell metadata (position, sequence) is shared by every layer. The stock
state API serializes whole sequences and has no notion of a block range.

## Decision

- **Two contexts over one model load.** `lower` executes and owns the K/V of
  blocks 0-17; `upper` of blocks 18-29 plus the final norm and head. Each keeps
  its own cell bookkeeping, so each sequence state is complete for the layers
  it owns and `llama_state_seq_get/set_data` serializes exactly those layers.
- **A recorded runtime patch, not a fork.**
  `riderless/api/native/patches/gemma4-layer-range.patch` adds
  `layer_begin`/`layer_end` to `llama_context_params`, filters the Gemma 4 KV
  cache to that range, runs only those blocks, takes residual input through
  `llama_batch.embd` when `layer_begin > 0`, and exposes the raw residual after
  the last executed block through the existing `layer_inp` capture. With the
  defaults the stock graph is unchanged. The patched base is built separately
  (`build_base_runtime.py --patch`) and its manifest records the patch hash.
- **Fixed captures per context.** Lower always captures H18, upper always
  captures H30 and the head input. Create, promote and every branch therefore
  run identical graphs, which is what the restore-identity gate relies on.
- **Host-side immutable snapshots, resident fast path.** A snapshot is its
  serialized per-range sequence states plus H18/H30. A branch reuses the
  contexts when they still hold the parent and otherwise restores it from host
  bytes; every branch trims back to the parent afterwards.
- **No early readout at block 18.** No early head is registered for the new
  prompt profile, so a block-18 readout is refused rather than substituted.

## Consequences

- The split profile is numerically distinct from the stock graph (different
  graph partitioning and fusion). Its qualification against the stock graph
  (`scripts/riderless/qualify_snapshots.py`) is a gate. A failure keeps the
  profile experimental.
- Only text-only Gemma 4 with no shared-KV layers, per-layer inputs or nextn
  layers is accepted. The worker and the patch both check this.
- The worker links a different runtime from the v1 worker and ships as its own
  bundle (`riderless/api/native/snapshots/build.py`).
